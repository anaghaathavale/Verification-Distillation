import numpy as np
import tensorflow as tf


from RobustMockTeacher import MockNeuralNetwork

#loss function (same as the one in Definition 1.3 (form Shao et al) in the overleaf document)
#y_true is the true label (from training dataset) 
#y_pred_S and y_pred_T are the predictions of the student and teacher models respectively
def LCE(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def LKL(y_pred_S, y_pred_T, temperature=1.0):
    y_pred_S /= temperature
    y_pred_T /= temperature
    return tf.keras.losses.KLDivergence(y_pred_S, y_pred_T)

def LGAD(x, y_true, y_pred_S, y_pred_T, lambda_CE, lambda_KL, lambda_GAD, temperature=1.0):
    # Cross-entropy loss
    CE_loss = LCE(y_true, y_pred_S)
    
    # KL-divergence loss
    KL_loss = temperature**2 * LKL(y_pred_S / temperature, y_pred_T / temperature)
    
    # Compute gradients of CE loss w.r.t. input x for both student and teacher models
    with tf.GradientTape() as tape_S:
        tape_S.watch(x)
        CE_loss_S = LCE(y_true, y_pred_S)
        grad_CE_S = tape_S.gradient(CE_loss_S, x)
        
    with tf.GradientTape() as tape_T:
        tape_T.watch(x)
        CE_loss_T = LCE(y_true, y_pred_T)
        grad_CE_T = tape_T.gradient(CE_loss_T, x)
        
    # Gradient discrepancy loss
    grad_discrepancy = tf.norm(grad_CE_S - grad_CE_T)
    
    # Combine losses
    LGAD_loss = lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy
    
    return LGAD_loss



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
    #student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    student_model.compile(optimizer='adam', loss=lambda y_true, y_pred: LGAD(inputs, y_true, y_pred, teacher_model(inputs), 
        lambda_CE, lambda_KL, lambda_GAD, temperature), metrics=['accuracy'])
    student_model.fit(synthetic_data, synthetic_labels, batch_size=batch_size, epochs=epochs)



if __name__ == "__main__":
    teacher = MockNeuralNetwork(42, 5, 1)
    knowledge_distillation(teacher,teacher,10**6,(5, ),1000,100)

