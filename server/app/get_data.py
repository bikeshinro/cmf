from cmflib import cmfquery
import pandas as pd

#This function fetches all the executions available in given mlmd
def get_executions(mlmdfilepath):
   query = cmfquery.CmfQuery(mlmdfilepath)
   names = query.get_pipeline_names()       #getting all pipeline names in mlmd
   for name in names:
        stages = query.get_pipeline_stages(name)
        df = pd.DataFrame()
        for stage in stages:
            executions = query.get_all_executions_in_stage(stage)
            df = pd.concat([df, executions], sort=True, ignore_index=True)
        return df

#This function fetches all the artifacts available in given mlmd
def get_artifacts(mlmdfilepath):
    query = cmfquery.CmfQuery(mlmdfilepath)
    names = query.get_pipeline_names()          #getting all pipeline names in mlmd
    identifiers = []
    for name in names:
        stages = query.get_pipeline_stages(name)
        for stage in stages:
            executions = query.get_all_executions_in_stage(stage)
            dict_executions = executions.to_dict("dict")  # converting it to dictionary
            identifiers.append(dict_executions['id'][0])
            # df = pd.concat([df, executions], sort=True, ignore_index=True)

    name = []
    url = []
    df = pd.DataFrame()
    for identifier in identifiers:
        get_artifacts = query.get_all_artifacts_for_execution(identifier)  # getting all artifacts
        artifacts_dict = get_artifacts.to_dict('dict')  # converting it to dictionary
        df = pd.concat([df, get_artifacts], sort=True, ignore_index=True)
    print(df.keys())
    # df.drop(['Commit','avg_prec','git_repo','last_update_time_since_epoch','metrics_name','model_framework','model_name','model_type','roc_auc','user-metadata1'],axis=1,inplace=True)
    return df



