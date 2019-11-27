import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

## If you want to compare Batch Normalization with Layer Normalization make 1
compareBNLN = 0

if compareBNLN == 1:
    filepath = '/media/nancy/D/Nancys_Diary/rethinking-bnn/zoo/cifar10_250epochs'
else:
    filepath = '/media/nancy/D/Nancys_Diary/rethinking-bnn/zoo/cifar10/binaryvgg'

if compareBNLN == 0 :
    fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(20,15))
    fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(20,15))
    fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(20,15))
    fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(20,15))
    for i in os.listdir(filepath):
        # print(i)
        if '1' in os.listdir(os.path.join(filepath, i)):
            configfilepath = os.path.join(filepath, i, '1/config.json')
        else:
            configfilepath = os.path.join(filepath, i, '2/config.json')
    
        with open(configfilepath) as f:
            d = json.load(f)
            # print(d)
            # print('Dataset: {}'.format(d["dataset"]))
            # print('Epochs:{}'.format(d['epochs']))
            # print('input_quantizer: {}'.format(d['input_quantizer']))
            # print('kernel_constraint: {}'.format(d['kernel_constraint']))
            # print('kernel_quantizer: {}'.format(d['kernel_quantizer']))
            # print('gamma: {}'.format(d['opt_param:gamma']))
            gamma = d['opt_param:gamma']
            # print('threshold: {}'.format(d['opt_param:threshold']))
            threshold = d['opt_param:threshold']
            # print('Batch/Layer normalization: {}'.format(d['norm_layer']))
            Normalization = d['norm_layer']

        if Normalization=='LayerNormalization':
            steps = []
            cat_acc = []
            cat_loss = []
            val_acc = []
            val_loss = []
            if threshold == 1e-8:
                print(Normalization, i, gamma, threshold)
                if '1' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
                    with open(metricfilepath1) as f:
                        d = json.load(f)
                        steps = d['categorical_accuracy']['steps']
                        cat_acc = d['categorical_accuracy']['values']
                        cat_loss = d['loss']['values']
                        val_acc = d['val_categorical_accuracy']['values']
                        val_loss = d['val_loss']['values']

                if '2' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath2 = os.path.join(filepath, i, '2/metrics.json')
                    with open(metricfilepath2) as f:
                        d = json.load(f)
                        # steps = np.concatenate(steps, d['categorical_accuracy']['steps'])
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']

                if '3' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath3 = os.path.join(filepath, i, '3/metrics.json')
                    with open(metricfilepath3) as f:
                        d = json.load(f)
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']
            
                fig1.suptitle('LayerNorm: Categorical Accuracy (CA) and Categorical Loss (CL)')

                ax11.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CA, $\gamma$={}, $tau$={}'.format(gamma, threshold))
            
                # ax1.plot(steps, val_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='Validation accuracy')
                ax11.legend(loc=1)
                plt.ylabel('Accuracy')
                plt.xlabel('Timestep')
                ax11.grid()

                ax12.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CL, $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                # ax2.plot(steps, val_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='VL, gamma threshold' )
                ax12.legend(loc=1)
                plt.ylabel('Loss')
                plt.xlabel('Timestep')
                ax12.grid()

            fig1.savefig("AccuracyLoss_LN_fixedtau.pdf", format = 'pdf', bbox_inches = 'tight' )

            if gamma == 1e-4:
                print(Normalization, i, gamma, threshold)
                if '1' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
                    with open(metricfilepath1) as f:
                        d = json.load(f)
                        steps = d['categorical_accuracy']['steps']
                        cat_acc = d['categorical_accuracy']['values']
                        cat_loss = d['loss']['values']
                        val_acc = d['val_categorical_accuracy']['values']
                        val_loss = d['val_loss']['values']

                if '2' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath2 = os.path.join(filepath, i, '2/metrics.json')
                    with open(metricfilepath2) as f:
                        d = json.load(f)
                        # steps = np.concatenate(steps, d['categorical_accuracy']['steps'])
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']

                if '3' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath3 = os.path.join(filepath, i, '3/metrics.json')
                    with open(metricfilepath3) as f:
                        d = json.load(f)
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']
            
                fig2.suptitle('LayerNorm: Categorical Accuracy (CA) and Categorical Loss (CL)')

                ax21.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CA,  $\gamma$={}, $tau$={}'.format(gamma, threshold))
            
                # ax1.plot(steps, val_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='Validation accuracy')
                ax21.legend(loc=1)
                plt.ylabel('Accuracy')
                plt.xlabel('Timestep')
                ax21.grid()

                ax22.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CL,  $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                # ax2.plot(steps, val_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='VL, gamma threshold' )
                ax22.legend(loc=1)
                plt.ylabel('Loss')
                plt.xlabel('Timestep')
                ax22.grid()

            fig2.savefig("AccuracyLoss_LN_fixedgamma.pdf", format = 'pdf', bbox_inches = 'tight' )


        if Normalization=='BatchNormalization':
            steps = []
            cat_acc = []
            cat_loss = []
            val_acc = []
            val_loss = []
            if threshold == 1e-8:
                print(Normalization, i, gamma, threshold)
                if '1' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
                    with open(metricfilepath1) as f:
                        d = json.load(f)
                        steps = d['categorical_accuracy']['steps']
                        cat_acc = d['categorical_accuracy']['values']
                        cat_loss = d['loss']['values']
                        val_acc = d['val_categorical_accuracy']['values']
                        val_loss = d['val_loss']['values']

                if '2' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath2 = os.path.join(filepath, i, '2/metrics.json')
                    with open(metricfilepath2) as f:
                        d = json.load(f)
                        # steps = np.concatenate(steps, d['categorical_accuracy']['steps'])
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']

                if '3' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath3 = os.path.join(filepath, i, '3/metrics.json')
                    with open(metricfilepath3) as f:
                        d = json.load(f)
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']
            
                fig3.suptitle('BatchNorm: Categorical Accuracy (CA) and Categorical Loss (CL)')

                ax31.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CA, $\gamma$={}, $tau$={}'.format(gamma, threshold))
            
                # ax1.plot(steps, val_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='Validation accuracy')
                ax31.legend(loc=1)
                plt.ylabel('Accuracy')
                plt.xlabel('Timestep')
                ax31.grid()

                ax32.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CL, $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                # ax2.plot(steps, val_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='VL, gamma threshold' )
                ax32.legend(loc=1)
                plt.ylabel('Loss')
                plt.xlabel('Timestep')
                ax32.grid()

            fig3.savefig("AccuracyLoss_BN_fixedtau.pdf", format = 'pdf', bbox_inches = 'tight' )

            if gamma == 1e-4:
                print(Normalization, i, gamma, threshold)
                if '1' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
                    with open(metricfilepath1) as f:
                        d = json.load(f)
                        steps = d['categorical_accuracy']['steps']
                        cat_acc = d['categorical_accuracy']['values']
                        cat_loss = d['loss']['values']
                        val_acc = d['val_categorical_accuracy']['values']
                        val_loss = d['val_loss']['values']

                if '2' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath2 = os.path.join(filepath, i, '2/metrics.json')
                    with open(metricfilepath2) as f:
                        d = json.load(f)
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']

                if '3' in os.listdir(os.path.join(filepath, i)):
                    metricfilepath3 = os.path.join(filepath, i, '3/metrics.json')
                    with open(metricfilepath3) as f:
                        d = json.load(f)
                        steps = steps + d['categorical_accuracy']['steps']
                        cat_acc = cat_acc + d['categorical_accuracy']['values']
                        cat_loss = cat_loss + d['loss']['values']
                        val_acc = val_acc + d['val_categorical_accuracy']['values']
                        val_loss = val_loss + d['val_loss']['values']
            
                fig4.suptitle('BatchNorm : Categorical Accuracy (CA) and Categorical Loss (CL)')

                ax41.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CA, $\gamma$={}, $tau$={}'.format(gamma, threshold))
            
                # ax1.plot(steps, val_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='Validation accuracy')
                ax41.legend(loc=1)
                plt.ylabel('Accuracy')
                plt.xlabel('Timestep')
                ax41.grid()

                ax42.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='CL, $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                # ax2.plot(steps, val_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='VL, gamma threshold' )
                ax42.legend(loc=1)
                plt.ylabel('Loss')
                plt.xlabel('Timestep')
                ax42.grid()

            fig4.savefig("AccuracyLoss_BN_fixedgamma.pdf", format = 'pdf', bbox_inches = 'tight' )

    plt.show()

else:
    fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(20,15))
    fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(20,15))

    for i in os.listdir(filepath):
        # print(i)
        if '1' in os.listdir(os.path.join(filepath, i)):
            configfilepath = os.path.join(filepath, i, '1/config.json')
        else:
            configfilepath = os.path.join(filepath, i, '2/config.json')
    
        with open(configfilepath) as f:
            d = json.load(f)
            # print(d)
            # print('Dataset: {}'.format(d["dataset"]))
            # print('Epochs:{}'.format(d['epochs']))
            # print('input_quantizer: {}'.format(d['input_quantizer']))
            # print('kernel_constraint: {}'.format(d['kernel_constraint']))
            # print('kernel_quantizer: {}'.format(d['kernel_quantizer']))
            # print('gamma: {}'.format(d['opt_param:gamma']))
            gamma = d['opt_param:gamma']
            # print('threshold: {}'.format(d['opt_param:threshold']))
            threshold = d['opt_param:threshold']
            # print('Batch/Layer normalization: {}'.format(d['norm_layer']))
            Normalization = d['norm_layer']

        steps = []
        cat_acc = []
        cat_loss = []
        val_acc = []
        val_loss = []
        if threshold == 1e-8:
            print(Normalization, i, gamma, threshold)
            if '1' in os.listdir(os.path.join(filepath, i)):
                metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
                with open(metricfilepath1) as f:
                    d = json.load(f)
                    steps = d['categorical_accuracy']['steps']
                    cat_acc = d['categorical_accuracy']['values']
                    cat_loss = d['loss']['values']
                    val_acc = d['val_categorical_accuracy']['values']
                    val_loss = d['val_loss']['values']

            if '2' in os.listdir(os.path.join(filepath, i)):
                metricfilepath2 = os.path.join(filepath, i, '2/metrics.json')
                with open(metricfilepath2) as f:
                    d = json.load(f)
                    # steps = np.concatenate(steps, d['categorical_accuracy']['steps'])
                    steps = steps + d['categorical_accuracy']['steps']
                    cat_acc = cat_acc + d['categorical_accuracy']['values']
                    cat_loss = cat_loss + d['loss']['values']
                    val_acc = val_acc + d['val_categorical_accuracy']['values']
                    val_loss = val_loss + d['val_loss']['values']

            if '3' in os.listdir(os.path.join(filepath, i)):
                metricfilepath3 = os.path.join(filepath, i, '3/metrics.json')
                with open(metricfilepath3) as f:
                    d = json.load(f)
                    steps = steps + d['categorical_accuracy']['steps']
                    cat_acc = cat_acc + d['categorical_accuracy']['values']
                    cat_loss = cat_loss + d['loss']['values']
                    val_acc = val_acc + d['val_categorical_accuracy']['values']
                    val_loss = val_loss + d['val_loss']['values']
        
            fig1.suptitle('LayerNorm: Categorical Accuracy (CA) and Categorical Loss (CL)')

            ax11.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='{}, CA, $\gamma$={}, $tau$={}'.format(Normalization, gamma, threshold))
        
            # ax1.plot(steps, val_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='Validation accuracy')
            ax11.legend(loc=1)
            plt.ylabel('Accuracy')
            plt.xlabel('Timestep')
            ax11.grid()

            ax12.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='{}, CL, $\gamma$={}, $tau$={}'.format(Normalization, gamma, threshold) )
            # ax2.plot(steps, val_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='VL, gamma threshold' )
            ax12.legend(loc=1)
            plt.ylabel('Loss')
            plt.xlabel('Timestep')
            ax12.grid()

        fig1.savefig("AccuracyLoss_compareBNLN_fixedtau.pdf", format = 'pdf', bbox_inches = 'tight' )

        if gamma == 1e-4:
            print(Normalization, i, gamma, threshold)
            if '1' in os.listdir(os.path.join(filepath, i)):
                metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
                with open(metricfilepath1) as f:
                    d = json.load(f)
                    steps = d['categorical_accuracy']['steps']
                    cat_acc = d['categorical_accuracy']['values']
                    cat_loss = d['loss']['values']
                    val_acc = d['val_categorical_accuracy']['values']
                    val_loss = d['val_loss']['values']

            if '2' in os.listdir(os.path.join(filepath, i)):
                metricfilepath2 = os.path.join(filepath, i, '2/metrics.json')
                with open(metricfilepath2) as f:
                    d = json.load(f)
                    # steps = np.concatenate(steps, d['categorical_accuracy']['steps'])
                    steps = steps + d['categorical_accuracy']['steps']
                    cat_acc = cat_acc + d['categorical_accuracy']['values']
                    cat_loss = cat_loss + d['loss']['values']
                    val_acc = val_acc + d['val_categorical_accuracy']['values']
                    val_loss = val_loss + d['val_loss']['values']

            if '3' in os.listdir(os.path.join(filepath, i)):
                metricfilepath3 = os.path.join(filepath, i, '3/metrics.json')
                with open(metricfilepath3) as f:
                    d = json.load(f)
                    steps = steps + d['categorical_accuracy']['steps']
                    cat_acc = cat_acc + d['categorical_accuracy']['values']
                    cat_loss = cat_loss + d['loss']['values']
                    val_acc = val_acc + d['val_categorical_accuracy']['values']
                    val_loss = val_loss + d['val_loss']['values']
        
            fig2.suptitle('LayerNorm: Categorical Accuracy (CA) and Categorical Loss (CL)')

            ax21.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='{}, CA,  $\gamma$={}, $tau$={}'.format(Normalization, gamma, threshold))
        
            # ax1.plot(steps, val_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='Validation accuracy')
            ax21.legend(loc=1)
            plt.ylabel('Accuracy')
            plt.xlabel('Timestep')
            ax21.grid()

            ax22.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='{}, CL,  $\gamma$={}, $tau$={}'.format(Normalization, gamma, threshold) )
            # ax2.plot(steps, val_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1,label ='VL, gamma threshold' )
            ax22.legend(loc=1)
            plt.ylabel('Loss')
            plt.xlabel('Timestep')
            ax22.grid()

        fig2.savefig("AccuracyLoss_compareBNLN_fixedgamma.pdf", format = 'pdf', bbox_inches = 'tight' )


    plt.show()
