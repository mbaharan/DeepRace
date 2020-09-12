import yaml
# Batch norm fusion
from pytorch_bn_fusion.bn_fusion_tcn import fuse_bn_sequential
import torch

# Distiller Quantization
import distiller as ds
from distiller.data_loggers import collector_context
from distiller import file_config
from utility.helpers import resume_checkpoint, load_checkpoint, load_checkpoint_post, AverageMeter, CheckpointSaver, get_outdir, init_xavier
from collections import OrderedDict

class Quantize():
    def __init__(self, args, model, optimizer, testloader, loss_fn):
        self._bit = args.bit
        self._epoch = args.epochs
        self._QAT = args.QAT
        self._calibrate = args.calibrate
        self._post_training = args.qe_config_file
        self._QAT_yaml = './quantization_configs/quant_aware_linear.yaml'
        self._post_yaml = './quantization_configs/quant_post_linear.yaml'
        self._config = self.config()
        self._output_dir = args.train_path
        self._model_path = '{}/model_best.pth'.format(self._output_dir)
        self._model = model
        self._compression_scheduler = None
        self._optimizer = optimizer
        self._args = args
        self._testloader = testloader
        self._loss_fn = loss_fn

        if self._QAT:
            self.QAT()
        elif self._calibrate:
            self.calibrate()
        elif self._post_training:
            self.post_training()
        else:
            raise MissingQuantizationFunctionError(self)

    def config(self):

        if (self._QAT or self._calibrate) and not self._post_training:
            with open(self._QAT_yaml) as f:
                list_doc = yaml.safe_load(f)

                list_doc['policies'][0]['ending_epoch'] = self._epoch
                list_doc['quantizers']['linear_quantizer']['bits_activations'] = self._bit
                list_doc['quantizers']['linear_quantizer']['bits_weights'] = self._bit
                list_doc['quantizers']['linear_quantizer']['bits_bias'] = self._bit
                

                with open(self._QAT_yaml, 'w') as f:
                    yaml.safe_dump(list_doc, f, default_flow_style=False, sort_keys=False)
        
                f.close()
                
                return self._QAT_yaml

        elif self._post_training:
                quant_stats_path = '{}/qe_stats/quantization_stats.yaml'.format(self._output_dir)
                self._args.qe_config_file = self._post_yaml
                with open(self._post_yaml) as f:
                    list_doc = yaml.load(f)
                    list_doc['quantizers']['linear_quantizer']['model_activation_stats'] = quant_stats_path
                    with open(self._post_yaml, 'w') as f:
                        yaml.dump(list_doc, f, default_flow_style=False)
                print('-> Pointing post quantization YAMl to: {}'.format(quant_stats_path))
                f.close()

                return self.post_yaml
        else:
            raise InvalidQuantizationConfigError(self)


    def QAT(self):

        print("-> Loading from checkpoint for Online Quantization...")
        load_checkpoint(self._model, self._model_path)

        # Fuse BN layers
        if self._args.fuse_bn:
            print("-> Fusing BN to conv...")
            self._model.eval()
            self._model = fuse_bn_sequential(self._model)
            self._model.train()

        print("-> Preparing model for quantization-aware training...")
        self._compression_scheduler = file_config(self._model, self._optimizer, self._config, None)

        self._args.compress = self._config
        return self._args, self._model, self._compression_scheduler
        
    def calibrate(self):
        print("-> Preparing model for calibration...")

        if self._args.batch_size <= 128:
            print("**-> You can set your batch size higher if able (currently: {}), model will be in eval()<-**".format(self._args.batch_size))
            print('-> Setting batch size to 128 (default)')
            self._args.batch_size = 128

        if self._args.fuse_bn:
            print("-> Fusing BN to conv...")
            self._model.eval()
            self._model = fuse_bn_sequential(self._model)
            self._model.train()

        self._compression_scheduler = file_config(self._model, self._optimizer, self._config, None)

        # Load Quantized model
        print('+ Loading checkpoint from: {}/model_best.pth'.format(self._args.quantize_path))
        #load_checkpoint(model, model_path)
        same_keys, new_state_dict = load_checkpoint_post(self._model, self._model_path)

        base_bone = OrderedDict()
        for key in same_keys:
            base_bone[key] = new_state_dict[key]
        
        self._model.load_state_dict(base_bone)

        ds.utils.assign_layer_fq_names(self._model)
        collector = ds.data_loggers.QuantCalibrationStatsCollector(self._model)
        
        print("=> Validating...")
        with collector_context(collector):
            validate(self._model, self._testloader, self._loss_fn)


        qe_path = os.path.join(self.args._quantize_path, 'qe_stats')
        if not os.path.isdir(qe_path):
            os.mkdir(qe_path)
        yaml_path = os.path.join(qe_path, 'quantization_stats.yaml')
        collector.save(yaml_path)
        print("Quantization statics is saved at {}".format(yaml_path))
        return

    def post_training(self):
        print("-> Preparing model for Post training range-based quantization...")
        self._args.qe_config_file = self._post_training

        if self._args.fuse_bn:
            print("-> Fusing BN to conv...")
            self._model.eval()
            self._model = fuse_bn_sequential(self._model)
            self._model.train()

        print('+ Loading checkpoint from: {}'.format(self._model_path))
        same_keys, new_state_dict = load_checkpoint_post(self._model, self._model_path)

        base_bone = OrderedDict()
        for key in same_keys:
            base_bone[key] = new_state_dict[key]
        
        print("-> Loading new state dictionary...")
        self._model.load_state_dict(base_bone)

        print("-> Preparing quantizer")
        quantizer = ds.quantization.PostTrainLinearQuantizer.from_args(self.model, self._args)
        print("-> Dummy input")
        dummy_input = torch.rand((1, self._args.nhid)).cuda()
        
        quantizer.prepare_model(dummy_input)


class QuantizationError(Exception):
    pass

class MissingQuantizationFunctionError(QuantizationError):
    def __init__(self, options):
        self.message = 'Missing quantization option(s) - QAT: {}, calibrate: {}, post_training: {}. Please select one'.format(options._QAT, options._calibrate, options._post_training)
        super().__init__(self.message)

class InvalidQuantizationConfigError(QuantizationError):
    def __init__(self, options):
        self.message = 'Invalid quantization option(s) - QAT: {}, calibrate: {}, post_training: {}. Please select one'.format(options._QAT, options._calibrate, options._post_training)
        super().__init__(self.message)
