import argparse
import six, os, torch
from flask import Flask
from flask_cors import CORS
from flask_restful_swagger_2 import Api

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--op', type=str, default='train', choices=['train','train_bert','train_3', 'test','api','test_3', 'test_8','test_bert','test_bert_3','test_bert_8'], help='Choose operation')
    parser.add_argument('--target_gpu', type=str, default='0', choices=['0','1','2','m'], help='Choose target GPU')
    parser.add_argument('--ck_path', type=str, default=None, help='Write checkpoint path')
    parser.add_argument('--load_ck', type=str, default=None, help='Write checkpoint path')
    parser.add_argument('--port', type=str, default=None, help='Write api port')

    args = parser.parse_args()
    
    if isinstance(args.ck_path,six.string_types):
        args.ck_path = os.path.join("./checkpoints", args.ck_path)
        if not os.path.exists(args.ck_path):
            os.mkdir(args.ck_path)

    if isinstance(args.load_ck,six.string_types):
        args.load_ck = os.path.join("./checkpoints", args.load_ck)

    if args.op == 'train':
        from sources.run_classify import classify
        classify(do_train=True, args = args)
    elif args.op == 'train_bert':
         from sources.run_bert import classify
         classify(do_train=True, args = args)


    elif args.op == 'test':
        from sources.run_classify import classify
        classify(do_train=False, args = args)

    elif args.op == 'test_bert':
        from sources.run_bert import classify
        classify(do_train=False, args = args)


    elif args.op == 'api':
        import sources.api as A
        app = Flask(__name__)
        app.config['JSON_SORT_KEYS'] = False
        api = Api(app, title='API Template', api_version='0.0.1', api_spec_url='/swagger', host='localhost',
          description='API Template')
        cors = CORS(app, resources={r"*": {"origins": "*"}})
        A.args = args
        A.load_model()
        api.add_resource(A.API, "/")
        app.run(host='0.0.0.0', port=args.port,threaded = True,debug = True)





    
