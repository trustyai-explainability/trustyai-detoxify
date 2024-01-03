import argparse

from trustyai.detoxify import TMaRCo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--perc", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--target_feature", type=str, required=True)
    parser.add_argument("--content_feature", type=str, required=True)
    parser.add_argument("--td_columns", type=str, nargs='+', default=None)
    parser.add_argument("--model_prefix", type=str, default='g_')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    data_dir = args.data_dir
    perc = args.perc
    td_columns = args.td_columns

    target_feature = args.target_feature
    content_feature = args.content_feature
    model_prefix = args.model_prefix
    tmarco = TMaRCo()
    tmarco.train_models(perc=perc, dataset_name=dataset_name, expert_feature=target_feature, model_prefix=model_prefix,
                        data_dir=data_dir, content_feature=content_feature, td_columns=td_columns)
