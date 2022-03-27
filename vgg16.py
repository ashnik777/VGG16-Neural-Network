class bachcifar100vgg():
  def __init__(self):
    self.classes = 100
    self.weight_decay = 0.0005
    self.x_shape = [32,32,3]

    self.model = self.build()

  def build(self):
    model = Sequential()
    weight_decay = self.weight_decay

    model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=self.x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(64,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    


    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(256,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(256,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2,2)))
    

    model.add(Conv2D(512,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(512,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(512,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2,2)))
    

    model.add(Conv2D(512,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(512,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(512,(3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())

    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dropout(rate = 0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(self.classes))
    model.add(Activation('softmax'))

    return model
  
  def train(self, x, y, batch=50, epoch=10):

    self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    self.model.fit(x, y, batch, epoch)
    self.model.save_weights('cifar100vgg1.h5')

    return self.model
