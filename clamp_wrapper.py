# pylint: disable=missing-function-docstring
#!/usr/bin/env python

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pprint import pprint

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def read_data(source, id_field):
    if not os.path.isfile(source):
        raise ValueError(f'Input file does not exist: {source}')
    #
    logger.info(f'Reading {source}')
    if source.endswith('.xlsx'):
        data = pd.read_excel(source, engine='openpyxl')
    elif source.endswith('.csv'):
        data = pd.read_csv(source)
    elif source.endswith('.tsv'):
        data = pd.read_csv(source, sep='\t')
    else:
        raise ValueError('Input file should be in excel, csv, or tsv format.')

    logger.info(f'{data.shape[0]} records read')
    if id_field not in data:
        raise ValueError(
            f'ID field {id_field} does not exist. Please use option --id-field to specify one'
        )
    return data


def write_data(data, output_file, index):
    logger.info(f'Write to {output_file}')
    if output_file.endswith('.xlsx'):
        data.to_excel(output_file, index=index)
    elif output_file.endswith('.csv'):
        data.to_csv(output_file)
    elif output_file.endswith('.tsv'):
        data.to_csv(output_file, sep='\t')
    else:
        raise ValueError('Output file should be in excel, csv, or tsv format.')


def get_record(source_data, range, search, reg_search, is_empty, limit,
               id_field):
    if id_field not in source_data:
        raise ValueError(
            f'Expecting an id field from the data: {id_field} does not exist.')

    data = source_data.copy(deep=True)
    if range:
        if pd.api.types.is_integer_dtype(data[id_field]):
            data = data[(data[id_field] >= range[0])
                        & (data[id_field] <= range[1])]
        else:
            values = data[id_field].str.extract(
                '(\d+)', expand=False).astype(int)
            data = data[(values >= range[0]) & (values <= range[1])]
        logger.info(
            f'{data.shape[0]} records kept after restricting data from row {range[0]} to {range[1]}.'
        )
    if search:
        kept = None
        for words in search:
            # find rows that match the word
            rec_kept = None
            for word in words:
                mask = np.column_stack([
                    data[col].str.contains(
                        word, case=False, na=False, regex=False)
                    for col in data
                    if hasattr(data[col], 'str')
                ])
                if rec_kept is None:
                    rec_kept = mask.any(axis=1)
                else:
                    rec_kept = rec_kept & mask.any(axis=1)
            if kept is None:
                kept = rec_kept
            else:
                kept = kept | rec_kept
        data = data[kept]
        logger.info(
            f'{data.shape[0]} records kept after searching rows that matches regular expression {" OR ".join([" AND ".join(x) for x in search])}.'
        )

    if reg_search:
        kept = None
        for words in reg_search:
            # find rows that match the word
            rec_kept = None
            for word in words:
                mask = np.column_stack([
                    data[col].str.contains(word, na=False, regex=True)
                    for col in data
                    if hasattr(data[col], 'str')
                ])
                if rec_kept is None:
                    rec_kept = mask.any(axis=1)
                else:
                    rec_kept = rec_kept & mask.any(axis=1)

            if kept is None:
                kept = rec_kept
            else:
                kept = kept | rec_kept
        data = data[kept]
        logger.info(
            f'{data.shape[0]} records kept after searching rows with word {" OR ".join([" AND ".join(x) for x in reg_search])}.'
        )
    #
    if is_empty:
        if is_empty not in data:
            logger.warning(f'Column {is_empty} is not found in the data.')
        else:
            data = data[(data[is_empty].isnull()) | (data[is_empty] == '')]
            logger.info(
                f'{data.shape[0]} records kept after removing non-empty rows at column {is_empty}'
            )
    #
    if data.shape[0] == 0:
        logger.info(f'No processable record is found.')
        sys.exit(0)

    if limit and limit < data.shape[0]:
        data = data.head(n=limit)
        logger.info(f'Limiting to first {limit} records')
    return data


def execute_process(data, field, clamp_jar_file, clamp_license_file,
                    clamp_pipeline, umls_api_key, umls_index_dir, semantics,
                    id_field, input_dir, output_dir, dryrun):
    # write new files over
    for _, row in data.iterrows():
        if dryrun:
            continue
        input_file = os.path.join(input_dir, f"data_{row[id_field]}.txt")
        logger.info(f'Exporting to {input_file}')
        try:
            with open(input_file, 'w') as ifile:
                ifile.write(row[field])
        except UnicodeEncodeError:
            with open(input_file, 'wb') as ifile:
                ifile.write(row[field].encode('utf8'))
    #
    # CLAMP command
    if not semantics:
        raise ValueError(
            'Please specify at least one semantics in the format of SEMANTIC=ASSERTION using parameter --semantics'
        )
    if not umls_api_key:
        raise ValueError(
            'Please specify your UMLS api key with option --umls-api-key')
    if not clamp_pipeline:
        raise ValueError(
            'Please specify a pipeline (a .jar file) with option --clamp-pipeline'
        )
    if not os.path.isdir(umls_index_dir):
        raise ValueError(
            f'UMLS index directory {umls_index_dir} does not exist.')
    cmd = [
        'java', f'-DCLAMPLicenceFile={clamp_license_file}', '-Xmx3g', '-cp',
        clamp_jar_file, 'edu.uth.clamp.nlp.main.PipelineMain', '-i', input_dir,
        '-o', output_dir, '-p', clamp_pipeline, '-A', umls_api_key, '-I',
        umls_index_dir
    ]
    #
    if not dryrun:
        logging.info(" ".join(cmd))
        subprocess.call(cmd)
    res = {}
    for _, row in data.iterrows():
        # process output file
        res[row[id_field]] = {x: 0 for x in semantics}
        if dryrun:
            continue
        output_file = os.path.join(output_dir, f"data_{row[id_field]}.txt")
        if not os.path.exists(output_file):
            raise RuntimeError(f'Failed to locate output file {output_file}')
        with open(output_file) as ofile:
            n_semantics = 0
            n_recorded = 0
            ignored_semantics = set()
            for line in ofile:
                # find lines such as
                # NamedEntity	496	514	semantic=problem	assertion=present	cui=C2939420	ne=metastatic disease
                #
                if not line.startswith('NamedEntity'):
                    continue
                n_semantics += 1
                fields = line.split()
                if not fields[3].startswith(
                        'semantic=') or not fields[4].startswith('assertion='):
                    logger.warning(f'Unrecognizable output: {line.strip()}')
                    continue
                val = fields[3][9:] + '=' + fields[4][10:]
                if val in semantics:
                    res[row[id_field]][val] += 1
                    n_recorded += 1
                else:
                    ignored_semantics.add(val)
            if n_semantics == 0:
                logger.info(
                    f'{output_file}: {n_semantics} semantic identified. {n_recorded} recorded, {", ".join(ignored_semantics) if ignored_semantics else "none"} ignored.'
                )
            else:
                logger.warning(
                    f'{output_file}: {n_semantics} semantic identified. {n_recorded} recorded, {", ".join(ignored_semantics) if ignored_semantics else "none"} ignored.'
                )
    return res


def process_data(data, field, clamp_jar_file, clamp_license_file,
                 clamp_pipeline, clamp_project_dir, umls_api_key,
                 umls_index_dir, semantics, id_field, dryrun):
    if not clamp_project_dir:
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                return execute_process(data, field, clamp_jar_file,
                                       clamp_license_file, clamp_pipeline,
                                       umls_api_key, umls_index_dir, semantics,
                                       id_field, input_dir, output_dir, dryrun)

    input_dir = os.path.join(
        os.path.expanduser(clamp_project_dir), 'Data', 'Input')
    output_dir = os.path.join(
        os.path.expanduser(clamp_project_dir), 'Data', 'Output')
    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        input_dir = os.path.join(os.path.expanduser(clamp_project_dir), 'input')
        if not os.path.isdir(input_dir):
            raise RuntimeError(
                f'Input directory Data/Input or input does not exist under {clamp_project_dir}.'
            )
        output_dir = os.path.join(
            os.path.expanduser(clamp_project_dir), 'output')
        if not os.path.isdir(output_dir):
            raise RuntimeError(
                f'Output directory Data/Output or output does not exist under {clamp_project_dir}.'
            )
    # clear data
    for file in os.scandir(input_dir):
        try:
            os.remove(file.path)
        except Exception:
            pass

    for file in os.scandir(output_dir):
        try:
            os.remove(file.path)
        except Exception:
            pass

    return execute_process(data, field, clamp_jar_file, clamp_license_file,
                           clamp_pipeline, umls_api_key, umls_index_dir,
                           semantics, id_field, input_dir, output_dir, dryrun)


def write_records(output_file, records, same_file, id_field):
    if same_file:
        logger.info(f'No need to update the existing file with no result.')
        return

    logger.info(f'Oututing selected samples to {output_file}.')

    if not os.path.isfile(output_file):
        logger.info(
            f'Writing {records.shape[0]} records to a new output file {output_file}'
        )
        write_data(records, output_file, index=False)
    else:
        old_data = read_data(output_file, id_field=id_field)
        logger.info(
            f'Appending or updating {records.shape[0]} records to a new output file {output_file}'
        )
        # remove existing records
        new_ids = set(records[id_field])
        new_data = pd.concat(
            [old_data[~old_data[id_field].isin(new_ids)], records],
            axis=0).sort_values(by=id_field)
        write_data(new_data, output_file, index=False)


def write_results(output_file, data, results, same_file, id_field):
    logger.info(
        f'Updating results from {len(results)} records in {output_file}.')

    for key in res.keys():
        if key not in data:
            logger.info(f'Adding column {key} to {output_file}')
            data.insert(2, key, [None for x in data[id_field]])
        col_idx = data.columns.get_loc(key)
        row_map = {v:k for k,v in data[id_field].to_dict().items()}
        for id, res in results.items():
            data.iloc[row_map[id], col_idx] = res[key]

    if same_file:
        logger.info(f'Updating original input file {output_file}')
        write_data(data, output_file, index=False)
    elif not os.path.isfile(output_file):
        logger.info(
            f'Writing {len(results)} records to a new output file {output_file}'
        )
        data = data[data[id_field].isin(results)]
        write_data(data, output_file, index=False)
    else:
        old_data = read_data(output_file, id_field=id_field)
        logger.info(
            f'Appending or updating {len(results)} records to a new output file {output_file}'
        )
        # remove existing records
        new_data = pd.concat([
            old_data[~old_data[id_field].isin(results)],
            data[data[id_field].isin(results)]
        ],
                             axis=0).sort_values(by=id_field)
        write_data(new_data, output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''Process NLP data from an excel file by calling CLAMP. This
command can be used as follows:

1. Search input file for fields. E.g.
    * Search by range of an ID field ('id')
        clamp_wrapper.py --input-file input.xlsx --range 1 10

    * Search by range of an alternative ID field
        clamp_wrapper.py --input-file input.xlsx --range 1 10 --id-field patient_id

    * Limit number of output fields
        clamp_wrapper.py --input-file input.xlsx --limit 10

    * Search fields that contains breast
        clamp_wrapper.py --input-file input.xlsx --search breast

    * Search fields that contains breast and cancer (order does not matter)
        clamp_wrapper.py --input-file input.xlsx --search breast cancer

    * Search fields that contains breast and cancer, or colon and cancer.
        clamp_wrapper.py --input-file input.xlsx --search breast cancer --search colon cancer

    * Search for records that have not been processed
        clamp_wrapper.py --input-file input.xlsx --is-empty 'problem=negated'

    There is another parameter --reg-search for searching by regular expression
    for advanced usages.

2. Output searched records to another file, for example,

        clamp_wrapper.py --input-file input.xlsx --search breast cancer --search colon cancer \\
            --output-file cancer.xlsx

   The output file can be used as input for processing by CLAMP.

3. Process by CLAMP, you will need to provide parameters

    * --process-field    the field to be searched (output as a text file to be processed by CLAMP)
    * --clamp-pipeline   the pipeline (.jar file) used to process the input files
    * --semantics SEMANTIC=ASSERTION ....
			 the semantics to search in the output files. The occurance frequency
                         will be outputed.

    E.g.
        clamp_wrapper.py --input-file input.xlsx --search breast cancer --search colon cancer \\
            --process-field impression --clamp-pipeline pipeline/clamp-ner.pipeline.jar \\
            --semantics problem=present problem=negated


4. Write output to output file. If --output-file is present, the record and results (with each semantics
   as a separate column) will be written to the output file.

   E.g.
        clamp_wrapper.py --input-file input.xlsx --search breast cancer --search colon cancer \\
            --process-field impression --clamp-pipeline pipeline/clamp-ner.pipeline.jar \\
            --semantics problem=present problem=negated \\
            --output=file input.xlsx

   In this particular case, the results will be write back to input file. Columns "problem=present" and
   "problem=negated" will be added if not already present.


Because the command line can be pretty line, you can write the parameters to a file (one line per word)
and read in through @filename. For example, with param.txt have the following content

--input-file
input.xlsx
--output-file
input.xlsx
--process-field
impression
--clamp-pipeline
pipeline/clamp-ner.pipeline.jar
--semantics
problem=present
problem=negated

The command can be simplied to

        clamp_wrapper.py --search breast cancer --search colon cancer @param.txt

''')
    parser.add_argument(
        '--input-file',
        required=True,
        help='''Path to an input file, which should be an excel or CSV file that has a numeric column
            named "id" with no duplicated value.''')
    parser.add_argument(
        '--id-field',
        default='id',
        help='''(Default to "id") Name of the id field''')
    parser.add_argument(
        '--range',
        nargs=2,
        type=int,
        help='''(Optional record filtering) Range of ids corresponding to the values of column "id" of the input
            file. Two values for lower and higher ids should be provided. If the column is not an integer type,
            it will be converted to integer by stripping all non-numeric characters'''
    )
    parser.add_argument(
        '--search',
        nargs='+',
        action='append',
        help='''(Optional record filtering) Find records that matches the search word as case-insensitive word. Multiple values
            can be specified and all of them should appear in the record ("--search abdomen contrast"
            for records that contain both "abdomen" and "contrast"). Use single quote to combine words
            (e.g. "--search 'abdomen contrast'") if the words should stay together. This parameter can be
            specified multiple times and records that match any of them will be processed (e.g.
            "--search abdomen contrast --search chest contrast" for records that contain "abdomen" and
            "contrast", or contains "chest" and "contrast")''')
    parser.add_argument(
        '--reg-search',
        nargs='+',
        action='append',
        help='''(Optional record filtering) Find records that matches the search word as regular expression. Multiple values can be
            specified and the parameter can be specified multiple times. See option --search for details.'''
    )
    parser.add_argument(
        '--is-empty',
        help='''(Optional record filtering) Find records that is empty for specified column'''
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='''(Optional record filtering) The maximum number of records to process.'''
    )
    parser.add_argument(
        '--process-field',
        help='''(Required for processing) Field in the excel file that will be exported and processed by
            CLAMP.''')
    parser.add_argument(
        '--clamp-jar-file',
        default='bin/clamp-nlp-1.6.6-jar-with-dependencies.jar',
        help='''(Default to bin/clamp-nlp-1.6.6-jar-with-dependencies.jar). Path to the CLAMP jar
             file such as /path/to/clamp-nlp-1.6.6-jar-with-dependencies.jar''')
    parser.add_argument(
        '--clamp-license-file',
        default='CLAMP.LICENSE',
        help='(Default to CLAMP.LICENSE). Path to the CLAMP license file.')
    parser.add_argument(
        '--clamp-pipeline',
        help='(Required for processing). Path to a jar file that contains the processing pipeline'
    )
    parser.add_argument(
        '--clamp-project-dir',
        help='''(Default to a temporary directory) Input folder for CLAMP directory, which should contain
            a directory Data/Input and a directory Data/Output, or a directory input and a directory
            output. If unspecified, a tempoary directory will be used, and removed after the data is processed.'''
    )
    parser.add_argument(
        '--umls-api-key',
        default='e0534990-f906-4120-87d3-92b4dabbc26b',
        help='(Required to run UMLS-based processing) UMLS API Key.')
    parser.add_argument(
        '--umls-index-dir',
        default='resource/umls_index/',
        help='(Default to resource/umls_index/) UMLS index directory, must exist before running CLAMP.'
    )
    parser.add_argument(
        '--batch-size',
        default=10000,
        type=int,
        help='''(Default to 10000). Number of records to be processed by CLAMP each time.'''
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='''If present, do not actually process data. The wrapper will read data, create fake result,
             and write output.'''
    )
    parser.add_argument(
        '--semantics',
        nargs='*',
        help='''(One or more values required for parsing output files) One or more values in the format
          of SEMANTIC=ASSERTION. This wrapper will look into the output files and search for rows with
          "NamedEntity", "semantic=SEMANTIC" and "assertion=ASSERTION", and output the count of such
          rows. Each value of this parameter will become a column in the resulting output file.'''
    )
    parser.add_argument(
        '--output-file',
        help='''(Required for extract output by CLAMP or write selected samples without processing,
            can be in excel or CSV format, determined by file extension). Write the results to an
            output file, which can be the same as the input file. If the file already exists, it will
            be updated with the results. New rows and new columns will be added to the file if needed.
            This option can be used to generate a subset of records if no CLAMP processing parameter
            is specified.''')

    args = parser.parse_args()
    data = read_data(args.input_file, args.id_field)
    records = get_record(data, args.range, args.search, args.reg_search,
                         args.is_empty, args.limit, args.id_field)
    if args.process_field:
        results = {}
        chunks = [
            args.batch_size for x in range(records.shape[0] // args.batch_size)
        ]
        for idx, chunk in enumerate(np.split(records, np.cumsum(chunks))):
            if chunk.shape[0] == 0:
                continue
            start = args.batch_size * idx + 1
            end = min(records.shape[0], args.batch_size * (idx + 1))
            logger.info(
                f'Processing {chunk.shape[0]} records in batch {idx + 1} ({start} - {end}) of {records.shape[0]} records.'
            )
            results.update(
                process_data(
                    chunk,
                    args.process_field,
                    args.clamp_jar_file,
                    args.clamp_license_file,
                    args.clamp_pipeline,
                    args.clamp_project_dir,
                    args.umls_api_key,
                    args.umls_index_dir,
                    args.semantics,
                    id_field=args.id_field,
                    dryrun=args.dryrun))
            if args.output_file:
                write_results(
                    args.output_file,
                    data,
                    results,
                    same_file=args.input_file == args.output_file,
                    id_field=args.id_field)

        if not args.output_file:
            for id, res in results.items():
                idx = list(np.where(records[args.id_field] == id))[0]
                rec = records.iloc[idx].to_dict()
                rec.update(res)
                pprint(rec)
            logging.info(
                f"{records.shape[0]} records with results outputted. Use option '--output-file' to save results to a file."
            )
    else:
        # no processing, just output
        if args.output_file:
            write_records(
                args.output_file,
                records,
                same_file=args.input_file == args.output_file,
                id_field=args.id_field)
        else:
            for _, rec in records.iterrows():
                pprint(rec.to_dict())

            logging.info(
                f"{records.shape[0]} records outputted. Use option '--process-fields' to process the data"
            )
    logger.info(f'Done')
