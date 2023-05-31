import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import sys
import traceback

from azure.storage.blob import BlobServiceClient
from cgmml.common.data_utilities.rgbd_matching import *


def main(db_host: str, db_user: str, db_pw: str, blob_conn_str: str, exec_path: str, logger, args):
    num_artifacts : int = args.num_artifacts if args.num_artifacts else -1
    dataset_type = args.dataset_type

    logger.write(f"Beginning main execution. Data Category: {args.data_category}, Dataset Type: {dataset_type}, Number of Artifacts: {num_artifacts}")

    # get scans from db and create dataframe
    database_repo = DatabaseRepo(db_host, db_user, db_pw)
    query_results, column_names = database_repo.get_scans(args.data_category, dataset_type, None, num_artifacts)
    logger.write("Retrieved {len(query_results)} scans from the database.")
    
    scans_df = PandaFactory.create_scans_data_frame(query_results, column_names, logger)
    logger.write("Created dataframe from scans, dataframe size: {len(scans_df)}")


    
    dataframe = scans_df
    logger.write("Set zscore")
    dataframe['zscore'] = dataframe.apply(PandaFactory.calculate_zscore, axis=1)
    logger.write("Set diagnosis")
    dataframe['diagnosis'] = dataframe.apply(PandaFactory.get_diagnosis,axis=1)
    logger.write("Set lhfa")
    dataframe['lhfa'] = dataframe.apply(PandaFactory.calculate_lhfa_zscore,axis =1)
    logger.write("Set diagnosis lhfa")
    dataframe['diagnosis_lhfa'] = dataframe.apply(PandaFactory.diagnosis_on_lhfa,axis=1)
    dataframe.rename(columns = {'diagnosis':'diagnosis_wfh','zscore':'zscore_wfh','lhfa':'zscore_lhfa'}, inplace = True)
    logger.write("....Done.")
    df_to_process = dataframe

    logger.write("Get 'no of person' data and merge it into df_to_process.")
    pose_number_data, column_names_number_pose = database_repo.get_number_persons_pose(args.workflow_id_pose, None)
    df_no_of_person = PandaFactory.create_number_persons_data_frame(pose_number_data, column_names_number_pose)
    df_no_of_person= df_no_of_person.drop_duplicates(subset='artifact_id', keep='last')
    logger.write("Merging number of person data into main dataframe")
    # Merge the two DataFrames on the 'artifact_id' column
    df_to_process = df_to_process.merge(df_no_of_person, on='artifact_id', suffixes=('', '_temp'))
    # Drop the temporary 'scan_id' column from the merged DataFrame
    df_to_process.drop(columns=['scan_id_temp'], inplace=True)
    logger.write(f"Entries after merge with nop data: {len(df_to_process)}")

    logger.write("get the 'pose_result' data and merge it into the main dataframe")
    pose_results, column_names_pose = database_repo.get_pose_result(args.workflow_id_pose, None)
    logger.write(f"got {len(pose_results)} Pose Results")
    df_pose_results = PandaFactory.create_pose_data_frame(pose_results, column_names_pose)
    logger.write("Pose Data Frame Created.")
    df_pose_results= df_pose_results.drop_duplicates(subset='artifact_id', keep='last')
    logger.write(f"Found {len(df_pose_results)} pose result entries, columns: {df_pose_results.columns}, df_proc entries: {len(df_to_process)}")
    # Merge the two DataFrames on the 'artifact_id' column
    df_to_process = df_to_process.merge(df_pose_results, on='artifact_id', suffixes=('', '_temp'))
    # Drop the temporary 'scan_id' column from the merged DataFrame
    df_to_process.drop(columns=['scan_id_temp'], inplace=True)
    logger.write(f"Entries after merge with pose data: {len(df_to_process)}")

    logger.write("Finalize df PreProcessing.")

   
    if dataset_type == 'rgbd':
         #Match depthmap and rgb to rgbd
        #it is necessary to remove certain rows first before rows are fused
        df_to_process = PandaFactory.filter_rgbd_data(df_to_process)

        df_to_process = df_to_process.drop('artifact', axis=1)
        df_to_process = df_to_process.drop('artifact_id', axis=1)
        df_to_process = df_to_process.drop('format_temp', axis=1)

        fused_artifacts_dicts = match_df_with_depth_and_image_artifacts(df_to_process)
        df_to_process = pd.DataFrame(fused_artifacts_dicts)

    ###download blobs
    BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(blob_conn_str)
    # Gather file_paths, Remove duplicates
    _file_paths = list(set(df_to_process['file_path'].tolist()))
    logger.write(f"Preparing to download {len(_file_paths)} files.")
    CONTAINER_NAME_SRC_SA = "cgm-result"
    NUM_THREADS = 64
    pool = ThreadPool(NUM_THREADS)
    results = pool.map(
        lambda full_name: BlobRepo.download_from_blob_storage(
            src=full_name,
            dest=f"{exec_path}scans/{full_name}",
            container=CONTAINER_NAME_SRC_SA,
            blob_client=BLOB_SERVICE_CLIENT
        ),
        _file_paths
    )
    logger.write("Done with download.")

    logger.write("Start creating Pickle Files.")
    ### Convert Dataframe to list of query_result_dicts
    logger.write("Start creating Pickle Files.")
    query_results_dicts = df_to_process.to_dict('records')

    ### Process
    input_dir = f"{exec_path}scans/"
    artifact_processor = ArtifactProcessor(input_dir, exec_path, dataset_type=dataset_type, should_rotate_rgb=True)


    #spark works well for DataBricks Clusters, alternatively you can use normal Multi Threading
    use_spark = True
    if use_spark:
        def map_fct(query_result_dict):
            return query_result_dict, artifact_processor.create_and_save_pickle(query_result_dict)
        def map_fct_rgb(query_result_dict):
            return query_result_dict, ImageFactory.process_rgb(query_result_dict, input_dir, exec_path)
        # Processing all artifacts at once
        rdd = spark.sparkContext.parallelize(query_results_dicts,48)
        print(rdd.getNumPartitions())
        if dataset_type == 'rgb':
            rdd_processed = rdd.map(map_fct_rgb)
        else:
            rdd_processed = rdd.map(map_fct)
            processed_dicts_and_fnames = rdd_processed.collect()
            print(processed_dicts_and_fnames[:3])    
    else:
        def process_artifact(artifact_dict: dict):
            if dataset_type == 'rgb':
                return (artifact_dict, ImageFactory.process_rgb(artifact_dict, input_dir, exec_path))
            else:
                return (artifact_dict, artifact_processor.create_and_save_pickle(artifact_dict))
        # Set the number of parallel workers
        num_workers = 32
        # Create a ThreadPoolExecutor instance and parallelize the processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed_dicts_and_fnames = list(executor.map(process_artifact, query_results_dicts))

    logger.write(f"All processed samples: {len(processed_dicts_and_fnames)} pickle files in folder {input_dir}.")
    # Select successfully processed
    processed_dicts_with_success = []
    processed_fnames_with_success = []
    for query_result_dict, fn in processed_dicts_and_fnames:
        if fn != '':
            processed_dicts_with_success.append(query_result_dict)
            processed_fnames_with_success.append(fn)
    # Update dataframe
    logger.write(f"Successfully pickled samples: {len(processed_fnames_with_success)}")

    if(args.upload_to_blob_storage):
        logger.write("Upload Blobs")
        BlobRepo.upload_to_blob_storage(args.upload_blob_conn_str, processed_dicts_with_success, args.dataset_type, args.data_category)
        logger.write("Upload Blobs Finished")

    logger.write("Main execution finished.")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')    
    parser.add_argument('--blob_conn_str', metavar='blob_conn_str', required=True, help='connection string for blob storage')
    parser.add_argument('--data_category', metavar='data_category', required=True, help='Can be Train or Test')
    parser.add_argument('--dataset_type', default='Train', metavar='dataset_type', required=True, help='Kind of Preprocessing: rgb, depth, rgbd')
    parser.add_argument('--db_host', metavar='db_host', required=True, help='address of db')
    parser.add_argument('--db_user', metavar='db_user', required=True, help='address of db')
    parser.add_argument('--db_pw', metavar='db_pw', required=True, help='address of db')
    parser.add_argument('--exec_path', metavar='exec_path', required=True, help='path from where to exec python script')
    parser.add_argument('--path_to_log', metavar='path_to_log', required=True, help='the path to workspace')
    parser.add_argument('--upload_to_blob_storage', metavar='upload_to_blob_storage', required=True, help='Upload to a blob storage afterwards, yes or no.')
    parser.add_argument('--upload_blob_conn_str', metavar='upload_blob_conn_str', required=True, help='connection string for blob storage upload')
    parser.add_argument('--workflow_id_pose', metavar='workflow_id_pose', required=True, help='Workflow Id used in Standing SQL Query')
    #optional params 
    parser.add_argument('--num_artifacts', metavar='num_artifacts', required=False, type=int, help='Maximum Number of entries taken from database')
    args = parser.parse_args()

    # Add the following line after parsing the arguments:
    sys.path.append(args.exec_path)
    from Repos.BlobRepo import BlobRepo
    from Factories.ImageFactory import ImageFactory
    from LoggerPipe import LoggerPipe
    from Repos.DatabaseRepo import DatabaseRepo
    from Factories.PandaFactory import PandaFactory
    from cgmml.common.data_utilities.mlpipeline_utils import ArtifactProcessor

    logger = LoggerPipe(args.path_to_log)
    
    try:
        main(args.db_host, args.db_user, args.db_pw, args.blob_conn_str, args.exec_path, logger, args)
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.write(f"Execution of Script failed: {e}\nStack trace:\n{stack_trace}")
        raise e

