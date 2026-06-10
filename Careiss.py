# =========================================================
# DENTAL CARIES DETECTION
# CNN vs EfficientNetB0 (FINAL MASTER VERSION - WITH LEARNING CURVES)
# =========================================================

import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_score,
    accuracy_score,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================================================
# REPRODUCIBILITY
# =========================================================
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# PATHS
# =========================================================
train_dir = "dataset/train"
test_dir = "dataset/test"

# =========================================================
# DATA GENERATION
# =========================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=SEED
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
    seed=SEED
)

# =========================================================
# CLASS WEIGHTS
# =========================================================
classes = train_generator.classes

class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)

class_weights = dict(zip(np.unique(classes), class_weights_values))

print("Class Weights:", class_weights)

# =========================================================
# EVALUATION FUNCTION
# =========================================================
def evaluate_model(model, test_gen):

    probs = model.predict(test_gen, verbose=0)
    preds = (probs > 0.5).astype(int).ravel()
    true = test_gen.classes

    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds)
    f1 = f1_score(true, preds)
    mcc = matthews_corrcoef(true, preds)

    cm = confusion_matrix(true, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    fpr, tpr, _ = roc_curve(true, probs.ravel())
    roc_auc = auc(fpr, tpr)

    print("\nClassification Report:\n")
    print(classification_report(true, preds))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }

# =========================================================
# CNN MODEL
# =========================================================
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn_callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
]

print("\n--- Training CNN ---")
cnn_history = cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=cnn_callbacks,
    class_weight=class_weights
)

# =========================================================
# CNN LEARNING CURVES
# =========================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title("CNN Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# EFFICIENTNETB0
# =========================================================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

efficient_model = Model(inputs=base_model.input, outputs=output)

efficient_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- Training EfficientNetB0 ---")
eff_history = efficient_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=cnn_callbacks,
    class_weight=class_weights
)

# =========================================================
# FINE TUNING
# =========================================================
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

efficient_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

finetune_callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
]

print("\n--- Fine Tuning EfficientNetB0 ---")
efficient_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=finetune_callbacks,
    class_weight=class_weights
)

# =========================================================
# EFFICIENTNET LEARNING CURVES
# =========================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(eff_history.history['accuracy'], label='Train Accuracy')
plt.plot(eff_history.history['val_accuracy'], label='Validation Accuracy')
plt.title("EfficientNet Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(eff_history.history['loss'], label='Train Loss')
plt.plot(eff_history.history['val_loss'], label='Validation Loss')
plt.title("EfficientNet Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# EVALUATION
# =========================================================
cnn_metrics = evaluate_model(cnn_model, test_generator)
efficient_metrics = evaluate_model(efficient_model, test_generator)

# =========================================================
# CONFUSION MATRIX
# =========================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(cnn_metrics["cm"], annot=True, fmt='d')
plt.title("CNN")

plt.subplot(1,2,2)
sns.heatmap(efficient_metrics["cm"], annot=True, fmt='d')
plt.title("EfficientNetB0")

plt.tight_layout()
plt.show()

# =========================================================
# ROC CURVE
# =========================================================
plt.figure(figsize=(8,6))

plt.plot(cnn_metrics["fpr"], cnn_metrics["tpr"],
         label=f'CNN AUC={cnn_metrics["auc"]:.4f}')

plt.plot(efficient_metrics["fpr"], efficient_metrics["tpr"],
         label=f'EfficientNet AUC={efficient_metrics["auc"]:.4f}')

plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Comparison")
plt.show()

# =========================================================
# COMPARISON TABLE
# =========================================================
comparison = pd.DataFrame({

    "Model": ["CNN", "EfficientNetB0"],

    "Accuracy": [
        cnn_metrics["accuracy"],
        efficient_metrics["accuracy"]
    ],

    "Precision": [
        cnn_metrics["precision"],
        efficient_metrics["precision"]
    ],

    "Sensitivity": [
        cnn_metrics["sensitivity"],
        efficient_metrics["sensitivity"]
    ],

    "Specificity": [
        cnn_metrics["specificity"],
        efficient_metrics["specificity"]
    ],

    "F1": [
        cnn_metrics["f1"],
        efficient_metrics["f1"]
    ],

    "MCC": [
        cnn_metrics["mcc"],
        efficient_metrics["mcc"]
    ],

    "AUC": [
        cnn_metrics["auc"],
        efficient_metrics["auc"]
    ]
})

print("\nFinal Comparison:\n")
print(comparison)

# =========================================================
# SAVE MODELS
# =========================================================
cnn_model.save("CNN_Dental_Caries.keras")
efficient_model.save("EfficientNet_Dental_Caries.keras")

comparison.to_csv("Model_Comparison.csv", index=False)

print("\nDONE - All outputs generated successfully!")
