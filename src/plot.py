import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


DenoisingAE = 0 ## 1 if Autoencoder graphs, 0 if image-classification


if DenoisingAE ==0:
    ## If you want to compare Batch Normalization with Layer Normalization make 1
    compareBNLN = 1

    if compareBNLN == 1:
        filepath = './../zoo/cifar10_250epochs'
    else:
        filepath = './../zoo/cifar10/binaryvgg'

    if compareBNLN == 0 :
        fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(7,2.5))
        fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(7,2.5))
        fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(7,2.5))
        fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(7,2.5))
        for i in os.listdir(filepath):
            # print(i)
            if '1' in os.listdir(os.path.join(filepath, i)):
                configfilepath = os.path.join(filepath, i, '1/config.json')
            else:
                configfilepath = os.path.join(filepath, i, '2/config.json')
        
            with open(configfilepath) as f:
                d = json.load(f)
                gamma = d['opt_param:gamma']
                threshold = d['opt_param:threshold']
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

                    ax11.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label =r'$\gamma$={}, $\tau$={}'.format(gamma, threshold))
                    ax11.set_ylabel('Categorical Accuracy',  labelpad=5)
                    ax11.set_xlabel('Timestep')
                    ax11.grid(True)

                    ax12.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)#,label ='$\gamma$={}, $tau$={}'.format(gamma, threshold) )
                    ax12.set_ylabel('Categorical Loss')
                    ax12.set_xlabel('Timestep')
                    ax12.grid(True)

                fig1.legend(loc=1)
                fig1.tight_layout()
                fig1.subplots_adjust(left=0.1, wspace=0.3, right=0.7)   
                fig1.savefig("./../results/AccuracyLossLNfixedtau.pdf", format = 'pdf', bbox_inches = 'tight' )

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
                
                    ax21.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label =r'$\gamma$={}, $\tau$={}'.format(gamma, threshold))
                    ax21.set_ylabel('Categorical Accuracy')
                    ax21.set_xlabel('Timestep')
                    ax21.grid(True)

                    ax22.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)#,label ='CL,  $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                    ax22.set_ylabel('Categorical Loss')
                    ax22.set_xlabel('Timestep')
                    ax22.grid(True)
                fig2.legend(loc=1)
                fig2.tight_layout()
                fig2.subplots_adjust(left=0.1, wspace=0.3, right=0.7)
                fig2.savefig("./../results/AccuracyLossLNfixedgamma.pdf", format = 'pdf', bbox_inches = 'tight' )


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
                

                    ax31.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label =r'$\gamma$={}, $\tau$={}'.format(gamma, threshold))
                    ax31.set_ylabel('Categorical Accuracy')
                    ax31.set_xlabel('Timestep')
                    ax31.grid(True)

                    ax32.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)#,label ='CL, $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                    ax32.set_ylabel('Categoricalv Loss')
                    ax32.set_xlabel('Timestep')
                    ax32.grid(True)
                fig3.legend(loc=1)
                fig3.tight_layout()
                fig3.subplots_adjust(left=0.1, wspace=0.3, right=0.7)
                fig3.savefig("./../results/AccuracyLossBNfixedtau.pdf", format = 'pdf', bbox_inches = 'tight' )

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
                
                    ax41.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label =r'$\gamma$={}, $\tau$={}'.format(gamma, threshold))
                    ax41.set_ylabel('Categorical Accuracy')
                    ax41.set_xlabel('Timestep')
                    ax41.grid(True)

                    ax42.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)#,label ='CL, $\gamma$={}, $tau$={}'.format(gamma, threshold) )
                    ax42.set_ylabel('Categorical Loss')
                    ax42.set_xlabel('Timestep')
                    ax42.grid(True)
                fig4.legend(loc=1)
                fig4.tight_layout()
                fig4.subplots_adjust(left=0.1, wspace=0.3, right=0.7)
                fig4.savefig("./../results/AccuracyLossBNfixedgamma.pdf", format = 'pdf', bbox_inches = 'tight' )

        plt.show()

    else:
        fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(6,2.5))

        for i in os.listdir(filepath):
            if '1' in os.listdir(os.path.join(filepath, i)):
                configfilepath = os.path.join(filepath, i, '1/config.json')
            else:
                configfilepath = os.path.join(filepath, i, '2/config.json')
        
            with open(configfilepath) as f:
                d = json.load(f)
                gamma = d['opt_param:gamma']
                threshold = d['opt_param:threshold']
                Normalization = d['norm_layer']

            steps = []
            cat_acc = []
            cat_loss = []
            val_acc = []
            val_loss = []

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

            ax11.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label ='{}'.format(Normalization))
            ax11.set_ylabel('Categorical Accuracy')
            ax11.set_xlabel('Timestep')
            ax11.grid(True)

            ax12.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)#,label =r'{}, $\gamma$={}, $\tau$={}'.format(Normalization, gamma, threshold) )
            plt.ylabel('Categorical Loss')
            plt.xlabel('Timestep')
            ax12.grid(True)

            fig1.legend(loc=1)
            fig1.tight_layout()
            fig1.subplots_adjust(left=0.1, wspace=0.3, right=0.85)
            fig1.savefig("./../results/AccuracyLosscompareBNLN.pdf", format = 'pdf', bbox_inches = 'tight' )

        plt.show()

else: 
    filepath = './../zoo/cifar10/denoisingAE/cifar10/autoencoder'
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
    for i in os.listdir(filepath):
        configfilepath = os.path.join(filepath, i, '1/config.json')
        
    steps = []
    cat_acc = []
    cat_loss = []
    val_acc = []
    val_loss = []
  
    if '1' in os.listdir(os.path.join(filepath, i)):
        metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
        with open(metricfilepath1) as f:
            d = json.load(f)
            steps = d['PSNR']['steps']
            cat_acc = d['PSNR']['values']
            cat_loss = d['loss']['values']
            val_acc = d['val_PSNR']['values']
            val_loss = d['val_loss']['values']
    ax1.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label ='AE w/ Adam')
    ax2.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)
    ax1.set_ylabel('Loss',  labelpad=5)
    ax2.set_ylabel('PSNR',  labelpad=5)
    ax1.set_xlabel('Timestep')
    ax2.set_xlabel('Timestep')
    ax1.grid(True) 
    ax2.grid(True)      

    filepath = './../zoo/cifar10/denoisingAE/bnn/cifar10/binaryae'
    
    for i in os.listdir(filepath):
        configfilepath = os.path.join(filepath, i, '1/config.json')
        
    with open(configfilepath) as f:
        d = json.load(f)
        Normalization = d['norm_layer']

    if Normalization=='BatchNormalization':
        steps = []
        cat_acc = []
        cat_loss = []
        val_acc = []
        val_loss = []
       
        print(Normalization)
        if '1' in os.listdir(os.path.join(filepath, i)):
            metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
            with open(metricfilepath1) as f:
                d = json.load(f)
                steps = d['PSNR']['steps']
                cat_acc = d['PSNR']['values']
                cat_loss = d['loss']['values']
                val_acc = d['val_PSNR']['values']
                val_loss = d['val_loss']['values']

        ax1.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label ='BAE w/ Adam')
        ax2.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)       
      
    filepath = './../zoo/cifar10/denoisingAE/bop/cifar10/binaryae'
    
    for i in os.listdir(filepath):
        configfilepath = os.path.join(filepath, i, '1/config.json')
        
    with open(configfilepath) as f:
        d = json.load(f)
        gamma = d['opt_param:gamma']
        threshold = d['opt_param:threshold']
        Normalization = d['norm_layer']

    if Normalization=='BatchNormalization':
        steps = []
        cat_acc = []
        cat_loss = []
        val_acc = []
        val_loss = []
       
        print(Normalization, gamma, threshold)
        if '1' in os.listdir(os.path.join(filepath, i)):
            metricfilepath1 = os.path.join(filepath, i, '1/metrics.json')
            with open(metricfilepath1) as f:
                d = json.load(f)
                steps = d['PSNR']['steps']
                cat_acc = d['PSNR']['values']
                cat_loss = d['loss']['values']
                val_acc = d['val_PSNR']['values']
                val_loss = d['val_loss']['values']

        ax1.plot(steps, cat_loss, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3,label ='BAE w/ BOP')
        ax2.plot(steps, cat_acc, ls='-', ms=5, markevery=1,alpha=0.8, linewidth=1.3)

        fig1.legend(loc=1)
        fig1.tight_layout()
        fig1.subplots_adjust(left=0.12, wspace=0.4, right=0.75)
        fig1.savefig("./../results/AEtradionalBNNBOP.pdf", format = 'pdf', bbox_inches = 'tight' )
        plt.show()