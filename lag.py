import pandas
import tqdm
import os
import itertools
import multiprocessing

from helpers import semver


WORKERS = 12  # 12 cores


def chunks(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def version_distance(base, target, releases):
    """
    Compute the distance between `base` and `target` (release rank)
    based on the number of patch, minor and major updates in between.
    
    We assume that `releases` (a subset of `df_releases`) is sorted by `RankByVersion`.
    """
    major, minor, patch = 0, 0, 0
    looking_for = 'Patch'
    
    for row in releases.itertuples():
        if row.RankByVersion <= base:
            continue
        if row.RankByVersion > target:
            break
        
        if looking_for == 'Patch':
            if row.ReleaseType == 'Patch' or row.ReleaseType == 'Misc':
                patch += 1
            else:
                looking_for = row.ReleaseType
        if looking_for == 'Minor':
            if row.ReleaseType == 'Minor':
                minor += 1
            elif row.ReleaseType == 'Major':
                looking_for = 'Major'
        if looking_for == 'Major' and row.ReleaseType == 'Major':
            major += 1
        
    return (major, minor, patch)


def compute_lags(releases, time, next_time, constraint):
    """
    Compute lag at `time` and `next_time`, assuming that given `constraint`
    should be evaluated wrt. to given set of package `releases` (a DataFrame that's a
    subset of `df_releases`.
    
    Return a 2-uple for each time, with:
     - highest release installable at `time`;
     - highest release missed at `time`;
     - oldest release missed at `time`;
     - version lag at `time`;
     - temporal lag at `time`;
     
    Return None if no release are installable.
    """
    
    releases = (
        releases
        .assign(
            # Tag them depending on their availability at both times
            AvailableAtNextTime=lambda d: d['ReleaseDate'] <= next_time,
            AvailableAtTime=lambda d: d['ReleaseDate'] <= time,
            
            # Tag the ones that are installable (ie. accepted by the constraint)
            Installable=lambda d: d['Release'].isin(semver(constraint, d['Release']))
        )
        [lambda d: d['AvailableAtNextTime']]
    )
    
    results = []
    
    # Handle computation at two times points
    for is_next in (False, True):
        available_label = 'AvailableAtTime' if not is_next else 'AvailableAtNextTime'
        
        # Find highest installable and annotate missed releases
        try:
            highest_installable = releases[lambda d: d['Installable'] & d[available_label]].iloc[-1]
            _missed_ix = releases[available_label] & (releases['RankByVersion'] > highest_installable['RankByVersion'])
        except IndexError:
            # None are installable, dismiss
            return None
            
        # Find highest missed and oldest missed (required to compute version and temporal lags)
        try:
            _missed = releases[_missed_ix]
            
            highest_missed = _missed.iloc[-1]
            first_missed = _missed.iloc[_missed['RankByDate'].values.argmin()]
            version_lag = version_distance(
                highest_installable['RankByVersion'],
                highest_missed['RankByVersion'],
                releases[lambda d: d[available_label]],
            )
            current_time = time if not is_next else next_time
            temporal_lag = current_time - first_missed['ReleaseDate']
            
            results.append((
                highest_installable['Release'],
                highest_missed['Release'],
                first_missed['Release'],
                version_lag,
                temporal_lag,
            ))
        except IndexError:
            results.append((highest_installable['Release'], None, None, (0, 0, 0), 0))
            
    return results


def _wrapper(args):
    package, release, dependency, constraint, time, next_time, all_releases = args
    lags = compute_lags(all_releases, time, next_time, constraint)
    if lags is None:
        return None
    else:
        at_t, at_next = lags
        return (
            (package, release, dependency, constraint, time)
            + at_t
            + (next_time,)
            + at_next
        )
        

if __name__ == '__main__':
    CENSOR_DATE = pandas.to_datetime('2018-01-01')
    
    # Load data
    print('Load releases')
    df_releases = pandas.read_csv(
        'data/releases.csv.gz',
        parse_dates=['ReleaseDate', 'NextReleaseDateByDate']
    )
    print('.. {} releases'.format(len(df_releases)))
    
    print('Load dependencies')
    df_dependencies = pandas.read_csv('data-raw/libio-dependencies.csv.gz')
    print('.. {} dependencies'.format(len(df_dependencies)))
    
    # Filter data
    LAST_UPDATE_AFTER = pandas.to_datetime('2017-01-01')
    MIN_UPDATES = 2
    
    print('Filter packages')
    packages = (
        df_releases
        [lambda d: (d['ReleaseDate'] >= LAST_UPDATE_AFTER) & (d['RankByDate'] >= MIN_UPDATES)]
        .drop_duplicates('Package')
        ['Package']
    )
    print('.. {} packages'.format(len(packages)))

    print('Filter dependencies & join dates')
    df_dependencies = (
        df_dependencies
        [lambda d: (d['Project'].isin(packages)) & (d['Dependency'].isin(packages))]
        .merge(
            df_releases[['Package', 'Release', 'ReleaseDate', 'NextReleaseDateByDate']],
            how='left',
            left_on=['Project', 'Release'],
            right_on=['Package', 'Release']
        )
    )
    print('.. {} dependencies'.format(len(df_dependencies)))
    
    print('Filter releases & set index')
    df_releases = df_releases[lambda d: d['Package'].isin(packages)]
    print('.. {} releases'.format(len(df_releases)))
    
    # Fast access to releases
    print('Group releases by package')
    df_releases = (
        df_releases
        .sort_values('RankByVersion')
        .set_index('Package')
        .sort_index(kind='mergesort')  # Stable sort
    )
    
    # Prepare output
    OUTPUT_PATH = 'data/lags.csv'
    OUTPUT_COLUMNS = [
        'Package', 'Release', 'Dependency', 'Constraint',
        'Time', 'HighestInstallableAtTime', 'HighestMissedAtTime', 'OldestMissedAtTime', 'VersionLagAtTime', 'TemporalLagAtTime',
        'NextTime', 'HighestInstallableAtNextTime', 'HighestMissedAtNextTime', 'OldestMissedAtNextTime', 'VersionLagAtNextTime', 'TemporalLagAtNextTime',
    ]
    
    if os.path.exists(OUTPUT_PATH):
        print('Load existing output file')
        df_output = pandas.read_csv(
            OUTPUT_PATH,
            usecols=OUTPUT_COLUMNS,
        )
        if len(df_output) > 0:
            # Resume job
            last_output = df_output.iloc[-1]
            last_processed = (
                df_dependencies
                [
                    lambda d:
                    (d['Project'] == last_output['Package'])
                    & (d['Release'] == last_output['Release'])
                    & (d['Dependency'] == last_output['Dependency'])
                    & (d['Constraint'] == last_output['Constraint'])
                ]
            ).iloc[0].name
            row_number = df_dependencies.index.get_loc(last_processed)
            
            print('Output contains {} items'.format(len(df_output)))
            print('Resume job at {}/{}'.format(row_number + 1, len(df_dependencies)))
            df_dependencies = df_dependencies.iloc[row_number + 1:]
    else:
        print('Create new output file')
        df_output = pandas.DataFrame(columns=OUTPUT_COLUMNS)
        df_output.to_csv(OUTPUT_PATH, index=False)
    del df_output  # No need for this dataframe
    
    # Do the job
    print('Start computation')
    chunk_size = 1000
    iterable = df_dependencies.itertuples()
        
    for chunk in tqdm.tqdm(chunks(iterable, chunk_size), total=len(df_dependencies) // chunk_size):
        temp_data = []
        
        # Create jobs
        jobs = []
        for row in chunk:
            if row is None:
                break
            
            releases = df_releases.loc[row.Dependency]
            if isinstance(releases, pandas.Series):
                releases = pandas.DataFrame(data=[releases])
                
            jobs.append((
                row.Project, row.Release, row.Dependency, row.Constraint,
                row.ReleaseDate,
                CENSOR_DATE if pandas.isnull(row.NextReleaseDateByDate) else row.NextReleaseDateByDate,
                releases,
            ))
        
        if len(jobs) == 0:
            print('No more job')
            continue
        
        with multiprocessing.Pool(processes=WORKERS) as pool:
            # Submit jobs
            results = pool.imap_unordered(_wrapper, jobs, 50)
            
            # Handle results
            for result in tqdm.tqdm(results, total=chunk_size, desc='process', leave=False):
                if not(result is None):
                    temp_data.append(result)
        
        # Append to dataframe and save
        df_temp = pandas.DataFrame(data=temp_data, columns=OUTPUT_COLUMNS)
        df_temp.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
