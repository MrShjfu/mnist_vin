# structure program:

    .
    ├── ...  
    ├── data                                    folder store some images of each epoches GAN training    
    ├── generator                               weight of generator model (gan)
    ├── weight_cnn                              weight cnn 
    ├── MNIST_NOTEBOOK.ipynb                    main method for run program     
    ├── README.md                  
    ├── gan_genearor.py
    │   │   ├── define_discriminator_model()     method defined discimninator(using for detect fake and real images)
    │   │   ├── define_generator_model()         method defined generator( try to create an real image from latent space)  
    │   │   ├── define_gan(g_model,d_model)      method defined GAN( from disciminator model and generator)  
    │   │   ├── train_gan()                      train GAN( try to create an real image from latent space)  
    ├── image_processing.py
    │   │   ├── datagen = ImageDataGenerator()   data augmentation by Image generator
    │   │   ├── get_light_model()                defined model CNN by keras of tensor
    │   │   ├── train_generator                  train model CNN with data augmentation method
    │   │   ├── train                            train model CNN by default data
    │   │   ├── define_discriminator_model()     method defined discimninator(using for detect fake and real images)
    └── ...

## Metrics evaluate
    - ACCURACY 

## SUMARY: 
    https://github.com/MrShjfu/mnist_vin/blob/master/MNIST_NOTEBOOK.ipynb
    - training on full data without Image Data Generator :TRAIN_ACC = 09974 VAL_ACC = 0.9924 TEST_ACC = 0.9917
    - training on 10% data without Data augmentation: note [OVERFITTING] TRAIN_ACC: ~1.0, VAL_ACC:0.1071 TEST_ACC:0.1135
    - training on 10% data with Data augmentation: note [OVERFITTING] TRAIN_ACC: ~1.0, VAL_ACC:0.1071 TEST_ACC:0.1135
  ### Loss =  sparse_categorical_crossentropy and Optimizer =
    # adamax best on   train: 0.9994 test: 0.9924
    # adam best on     train: 0.9949 test: 0.9664
    # adadelta best on train: 0.9992 test: 0.9916
    # sgd best on      train: 0.9998 test: 0.9908
   
  #### GAN
    # to fake data for training [simple model - need working more to update the model ]
    
    
