def quantize(args, model):
 
    if args.quantize:
            if args.bit == 4:
                if args.quant_aware or args.calibrate:
                    # For online quantization (Step 1)
                    quantization_aware_yaml = './quantization_configs/quant_aware_linear_4bit.yaml'
                    print('-> Loading quantization aware (4-bit): {}'.format(quantization_aware_yaml))
                elif args.qe_config_file:
                    # For post training quantization (Step 4?)
                    quantization_post_yaml = './quantization_configs/quant_post_linear_4bit.yaml'
                    args.qe_config_file = quantization_post_yaml
                    with open(quantization_post_yaml) as f:
                        list_doc = yaml.load(f)
                        list_doc['quantizers']['linear_quantizer']['model_activation_stats'] = quant_stats
                        with open(quantization_post_yaml, 'w') as f:
                            yaml.dump(list_doc, f, default_flow_style=False)
                    print('-> Pointing post quantization YAMl to: {}'.format(quant_stats))

            if args.bit == 8:
                if args.quant_aware or args.calibrate:
                    # For online quantization (Step 1)
                    quantization_aware_yaml = './quantization_configs/quant_aware_linear_8bit.yaml'
                    print('-> Loading quantization aware (8-bit): {}'.format(quantization_aware_yaml))
                elif args.qe_config_file:
                    # For post training quantization (Step 4)
                    quantization_post_yaml = './quantization_configs/quant_post_linear_8bit.yaml'
                    args.qe_config_file = quantization_post_yaml
                    print('-> Loading post quantization (8bit): {}'.format(quantization_post_yaml))
                    with open(quantization_post_yaml) as f:
                        list_doc = yaml.safe_load(f)
                        list_doc['quantizers']['linear_quantizer']['model_activation_stats'] = quant_stats
                        with open(quantization_post_yaml, 'w') as f:
                            yaml.dump(list_doc, f, default_flow_style=False)
                    print('-> Pointing post quantization YAMl to: {}'.format(quant_stats))

            if args.resume and not args.quant_aware and not args.calibrate:
                if args.fuse_bn:
                    print("-> Fusing BN to conv...")
                    model = fuse_bn_recursively(model)

                    print("-> Preparing model for quantization-aware training...")
                    compression_scheduler = file_config(model, optimizer, quantization_aware_yaml, None)

                    print("-> Resuming from checkpoint for Online Quantization...")
                    load_checkpoint(model, args.resume)

            if args.quant_aware and not args.qe_config_file and not args.calibrate:
                print("-> Loading from checkpoint for Online Quantization...")
                load_checkpoint(model, args.resume)

                # Fuse BN layers (step 1a)
                if args.fuse_bn:
                    print("-> Fusing BN to conv...")
                    model = fuse_bn_recursively(model)
                # Quantization-aware training (step 2)

                print("-> Preparing model for quantization-aware training...")
                compression_scheduler = file_config(model, optimizer, quantization_aware_yaml, None)

            if args.resume and args.calibrate:
                print("-> Preparing model for calibration...")
                
                if args.batch_size <= 128:
                    print("**-> You can set your batch size higher if able (currently: {}), model will be in eval()<-**".format(args.batch_size))

                if args.fuse_bn:
                    print("-> Fusing BN to conv...")
                    model = fuse_bn_recursively(model)
                    
                print("-> Compression...")
                compression_scheduler = file_config(model, optimizer, quantization_aware_yaml, None)

                print("-> Loading checkpoint...")
                load_checkpoint(model, args.resume)

            
            if args.resume and args.qe_config_file:
                print("-> Post training range-based quantization...")

                if args.fuse_bn:
                    print("-> Fusing BN to conv...")
                    model = fuse_bn_recursively(model)



                print("-> Loading from checkpoint...")
                same_keys, new_state_dict = load_checkpoint_post(model, args.load)

                base_bone = OrderedDict()
                for key in same_keys:
                    base_bone[key] = new_state_dict[key]
                
                print("-> Loading new state dictionary...")
                model.load_state_dict(base_bone)

                

                print("-> Preparing quantizer")
                quantizer = ds.quantization.PostTrainLinearQuantizer.from_args(model, args)
                print("-> Dummy input")
                dummy_input = torch.rand((1, 3, args.img_size, args.img_size)).cuda()
                
                quantizer.prepare_model(dummy_input)