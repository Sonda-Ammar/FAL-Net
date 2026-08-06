# train_ablation.py
import argparse
import os
import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from data_loader import prepare_train_val_data, prepare_test_data
from model_ablation import build_unet, LoRALayer

def main():
    parser = argparse.ArgumentParser(description='Train ablation study configuration')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank (0 to disable)')
    parser.add_argument('--ssa', action='store_true', help='Enable SSA')
    parser.add_argument('--no-ssa', action='store_false', dest='ssa', help='Disable SSA')
    parser.set_defaults(ssa=True)
    parser.add_argument('--multiscale', action='store_true', help='Enable Multi-scale fusion')
    parser.add_argument('--no-multiscale', action='store_false', dest='multiscale', help='Disable Multi-scale fusion')
    parser.set_defaults(multiscale=True)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--freeze', action='store_true', help='Freeze ResNet50 backbone')
    parser.add_argument('--val-split', type=float, default=0.15, help='Fraction of the train set used for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the train/val split')

    # --- Chemins du train set uniquement (le test set n'est jamais chargé ici) ---
    parser.add_argument('--real-dir-train', type=str, required=True, help='Path to training Live/real images')
    parser.add_argument('--fake-dir-train', type=str, required=True, help='Path to training Fake/spoof images')

    parser.add_argument('--real-dir-test', type=str, required=True, help='Path to testing Live/real images')
    parser.add_argument('--fake-dir-test', type=str, required=True, help='Path to testing Fake/spoof images')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Train/val issus d'un split interne du train set (le test set reste isole, cf. test_ablation.py)
    train_generator, val_generator = prepare_train_val_data(
        args.real_dir_train, args.fake_dir_train, val_split=args.val_split, seed=args.seed
    )

    # Build model
    input_shape = (224, 224, 3)
    num_classes = 2
    model = build_unet(input_shape, num_classes, lora_rank=args.rank, use_ssa=args.ssa, use_multiscale=args.multiscale, freeze_backbone=args.freeze)

    # Afficher le nombre de paramètres
    trainable_count = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])

    print(f"Total params: {trainable_count + non_trainable_count:,}")
    print(f"Trainable params: {trainable_count:,}")
    print(f"Non-trainable params: {non_trainable_count:,}")

    # Compile (lr et batch_size inchangés ici ; alignement prévu côté papier)
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    model_path = os.path.join(args.output, 'best_model.h5')
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    test_generator, all_labels_test = prepare_test_data(args.real_dir_test, args.fake_dir_test)
    # Train
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        verbose=1,
        callbacks=[checkpoint, lr_reducer, early_stopping]
    )

    # --- Sauvegarde des infos d'entrainement (utilisees ensuite par test_ablation.py) ---
    training_info = {
        'config': vars(args),
        'val_accuracy': float(max(history.history['val_accuracy'])),
        'val_loss': float(min(history.history['val_loss'])),
        'epochs_ran': len(history.history['loss']),
        'trainable_params': int(trainable_count),
        'total_params': int(trainable_count + non_trainable_count),
        'model_path': model_path,
    }

    training_info_path = os.path.join(args.output, 'training_info.json')
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=4)

    print(f"Training complete. Best model saved to {model_path}")
    print(f"Training info saved to {training_info_path}")
    print(f"Best val_accuracy: {training_info['val_accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
