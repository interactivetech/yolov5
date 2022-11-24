import train
from pathlib import PosixPath
import pandas as pd
import os
from utils.callbacks import Callbacks
from pathlib import Path

import determined as det

def plot_metrics(save_dir):
    df = pd.read_csv(os.path.join(save_dir,'results.csv'))
    row = df.iloc[-1]
    for k,v in row.to_dict().items():
            print(k.split(" "),v)
    return {k.split(" ")[-1]:v for k,v in row.to_dict().items()}
#     for ind,row in df.iterrows():
#         # print(ind
#         r = row.to_dict()
#         print("r: ",r)
#         print(r.keys())
#         '''
#         ToDo: Fix
#         r:  {'               epoch': 0.0, '      train/box_loss': 0.039896, '      train/obj_loss': 0.066715, '      train/cls_loss': 0.01431, '   metrics/precision': 0.76851, '      metrics/recall': 0.68981, '     metrics/mAP_0.5': 0.7774, 'metrics/mAP_0.5:0.95': 0.56153, '        val/box_loss': 0.034548, '        val/obj_loss': 0.033736, '        val/cls_loss': 0.008491, '               x/lr0': 0.0937, '               x/lr1': 0.0007, '               x/lr2': 0.0007}
# [2022-07-21T19:45:59.054675Z] f60cc478 || dict_keys(['               epoch', '      train/box_loss', '      train/obj_loss', '      train/cls_loss', '   metrics/precision', '      metrics/recall', '     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss', '        val/obj_loss', '        val/cls_loss', '               x/lr0', '               x/lr1', '               x/lr2'])
        
#         '''
#         # core_context.train.report_training_metrics(
#         #             steps_completed = int(r['               epoch']),metrics={"mAP":r['     metrics/mAP_0.5']}
#         #         )
#         for k,v in row.to_dict().items():
#             print(k,v)
    
def plot_ap_metrics_last(save_dir):
    p = sorted(list(Path(save_dir).glob("ap_class_results*.csv")))[-1:]# ignore last one, validation runs twice! at end
    print(len(p))
    df_ap = pd.DataFrame()
    frames = []
    for ind,i in enumerate(p):
        print(ind,i)
        d = pd.read_csv(i)
        d['epoch'] = ind
        # print(d[])
        # print(d.head())
        print(df_ap.shape)
        df_ap = df_ap.append([d],ignore_index=True)
    print("df_ap['maps'].mean(): ", df_ap['maps'].mean())
    # Values to track over time 'P', 'R', 'map50s', 'maps', 'epoch'
    for cl in df_ap['Class'].unique():
        print("mAP-{}: ".format(cl),cl,df_ap.query("Class == @cl")['maps'].tolist() )
        print("mAP@0.5-{}: ".format(cl),cl,df_ap.query("Class == @cl")['map50s'].tolist() )
        print("precision-{}: ".format(cl),cl,df_ap.query("Class == @cl")['P'].tolist() )
        print("recall-{}: ".format(cl),cl,df_ap.query("Class == @cl")['R'].tolist() )
    return df_ap.to_dict()

if __name__ == '__main__':

    info = det.get_cluster_info()
    assert info is not None, "this example only runs on cluster"
    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id

    hparams = info.trial.hparams
    print(hparams)
    for k,v in hparams.items():
        print(k,v)
    with det.core.init() as core_context:
        for op in core_context.searcher.operations():
            try:
                results = train.run(data=hparams["data_yaml"],
                        imgsz=hparams["imgsz"],
                        batch_size=hparams["batch-size"],
                        weights=hparams["weights"],
                        noautoanchor=hparams["noautoanchor"],
                        project=hparams["project"],
                        name=hparams["name"],
                        save_json=hparams["save_json"],
                        epochs=hparams["epochs"],
                        hyp=hparams["hyp"],
                        workers=hparams["workers"],
                        core_context=core_context)
                print("PLOTTING RESULTS...")
                train_json = plot_metrics(results.save_dir)
                print("train_json: 'metrics/mAP_0.5:0.95: ",train_json['metrics/mAP_0.5:0.95'])
                op.report_progress(1)
                op.report_completed(train_json['metrics/mAP_0.5:0.95'])
                val_json = plot_ap_metrics_last(results.save_dir)
            except Exception as e:
                print(e)
            # return
    # dir = results.save_dir