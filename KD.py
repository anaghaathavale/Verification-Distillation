import numpy as np
import tensorflow as tf

from RobustMockTeacher import MockNeuralNetwork


def knowledge_distillation(teacher_model, student_model, num_samples, input_shape, batch_size, epochs):
    # Compile teacher model
    teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Generate synthetic data using normal distribution
    synthetic_data = np.random.normal(size=(num_samples, *input_shape))
    
    # Get teacher predictions for synthetic data
    teacher_predictions = teacher_model.predict(synthetic_data)
    
    # Convert teacher predictions to labels
    synthetic_labels = np.argmax(teacher_predictions, axis=1)
    synthetic_labels = tf.keras.utils.to_categorical(synthetic_labels)
    
    # Train student model using synthetic data and teacher predictions
    student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    student_model.fit(synthetic_data, synthetic_labels, batch_size=batch_size, epochs=epochs)


if __name__ == "__main__":
    teacher = MockNeuralNetwork(42, 5, 1)
    knowledge_distillation(teacher,teacher,10**6,(5, ),1000,100)


# //todo: change loss function
# //try generating huge no. of num of samples
# //github pe daalna hai