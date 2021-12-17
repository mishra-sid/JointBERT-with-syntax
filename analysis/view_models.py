import json
import pandas as pd

def get_files(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.strip().endswith('results_test.json') and not line.strip().endswith('results_dev.json'): 
                continue
            yield line

def main():
    path = 'models.txt'

    files = get_files(path)

    results = []
    for fn in files:
        with open(fn) as f:
            if 'atis' in fn:
                data = 'atis'
            elif 'snips' in fn:
                data = 'snips'
            else:
                continue
            ex = json.loads(f.read())
            name = fn.split('/')[-3]
            seed_epochs = fn.split('/')[-2].split(',')
            epochs = seed_epochs[1][len('epochs='):]
            seed = seed_epochs[0][len('seed='):]
            split = fn.split('/')[-1].split('.')[0].split('_')[-1]
            ex['name'] = name
            ex['epochs'] = epochs
            ex['seed'] = seed
            ex['split'] = split
            results.append(ex)
    
    
    results = sorted(results, key=lambda x: (x['name'], x['epochs'], x['seed']))
    df = pd.DataFrame.from_dict(results)
    df['data'] = df['name'].apply(lambda x: x.split('.')[0])
    df['setting'] = df['name'].apply(lambda x: '.'.join(x.split('.')[1:]) if len(x.split('.')) > 1 else 'baseline')
    df = df.drop(columns=['name'])
    dev_df = df[df['split'] == 'dev']
    test_df = df[df['split'] == 'test'] 

    
    dev_df = dev_df.groupby(['data', 'setting', 'epochs']).agg(loss_mean=('loss', 'mean'), loss_std=('loss', 'std'), intent_acc=('intent_acc', 'mean'), slot_precision=('slot_precision', 'mean'), slot_recall=('slot_recall', 'mean'), slot_f1=('slot_f1', 'mean'), semantic_frame_acc=('semantic_frame_acc', 'mean')).reset_index()
    test_df = test_df.groupby(['data', 'setting', 'epochs']).agg(loss_mean=('loss', 'mean'), loss_std=('loss', 'std'), intent_acc=('intent_acc', 'mean'), slot_precision=('slot_precision', 'mean'), slot_recall=('slot_recall', 'mean'), slot_f1=('slot_f1', 'mean'), semantic_frame_acc=('semantic_frame_acc', 'mean')).reset_index() 
    
    print(dev_df)
    print(test_df)
    test_recs = []
    test_ref = {'snips': {}, 'atis': {}}
    for ind, x in dev_df.iterrows():
        data = x['data']
        setting = x['setting']
        epochs = x['epochs']
        dev_loss = x['loss_mean']
        dev_std = x['loss_std']
        if setting not in test_ref[data] or (setting in test_ref[data] and dev_loss < test_ref[data][setting][1]): 
            test_ref[data][setting] = (epochs, dev_loss, dev_std)

    loss_recs = []
    for data in test_ref:
        for setting in test_ref[data]:
            epochs, dev_loss, dev_loss_std = test_ref[data][setting]
            for ind, row in test_df.iterrows():
                if row['data'] == data and row['setting'] == setting and row['epochs'] == epochs:
                    test_recs.append({k: row[k] for k in ['data', 'setting', 'intent_acc', 'slot_precision', 'slot_recall', 'slot_f1', 'semantic_frame_acc']})
                    loss_recs.append({'data': data, 'setting': setting, 'dev_loss_mean': dev_loss, 'dev_loss_std': dev_loss_std, 'test_loss_mean': row['loss_mean'], 'test_loss_std': row['loss_std']})
                    break
    test_perf_df = pd.DataFrame.from_dict(test_recs)
    loss_recs_df = pd.DataFrame.from_dict(loss_recs)
    test_perf_df.set_index(['data', 'setting'], inplace=True)
    loss_recs_df.set_index(['data', 'setting'], inplace=True)
    #print(loss_recs_df.to_markdown())
    print((test_perf_df * 100).to_markdown())
        #df.to_csv(f'results_{data}.csv', index=False)

if __name__ == '__main__':
    main()

