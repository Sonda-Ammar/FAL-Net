# train_ablation.py
import argparse
import os
import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from data_loader import prepare_data
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
    parser.add_argument('--keep-model', action='store_true', help='Keep the best model .h5 file after evaluation')
    parser.add_argument('--freeze', action='store_true', help='Freeze ResNet50 backbone')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # Data paths (using Hi_Scan as default per train_ablation.py)
    real_dir_train = '../Fingerprint/2017/LivDet_Fingerprint_2017/Greenbit/Liveness_Only/LivenessImages'
    fake_dir_train = '../Fingerprint/2017/LivDet_Fingerprint_2017/Greenbit/Liveness_Only/spoof/'
    real_dir_test = '../Fingerprint/2017/LivDet_Fingerprint_2017/Greenbit/LiveDetVerification/VerificationImages/'
    fake_dir_test = '../Fingerprint/2017/LivDet_Fingerprint_2017/Greenbit/LiveDetVerification/SpoofVerification/'
    
    # Prepare data
    train_generator, test_generator, all_labels_test = prepare_data(real_dir_train, fake_dir_train, real_dir_test, fake_dir_test)
    
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

    # Compile
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
    
    # Train
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=test_generator,
        verbose=1,
        callbacks=[checkpoint, lr_reducer, early_stopping]
    )
    
    # Evaluate
    custom_objects = {'LoRALayer': LoRALayer}
    best_model = load_model(model_path, custom_objects=custom_objects)
    
    predictions = best_model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)
    
    true_labels = np.argmax(all_labels_test, axis=1) # 1D labels
    
    live_indices = np.where(true_labels == 1)[0]
    spoof_indices = np.where(true_labels == 0)[0]
    
    # Standard threshold (0.5)
    apcer = np.mean(predicted_labels[spoof_indices] == 1) * 100
    bpcer = np.mean(predicted_labels[live_indices] == 0) * 100
    ace = (apcer + bpcer) / 2
    
    results = {
        'config': vars(args),
        'apcer': float(apcer),
        'bpcer': float(bpcer),
        'ace': float(ace),
        'val_accuracy': float(max(history.history['val_accuracy'])),
        'trainable_params': int(trainable_count),
        'total_params': int(trainable_count + non_trainable_count)
    }

    # Advanced Metrics & DET Curve Data
    try:
        from sklearn.metrics import roc_curve
        y_scores = predictions[:, 1]
        fpr, tpr, thresholds = roc_curve(true_labels, y_scores)
        
        apcer_arr = fpr
        bpcer_arr = 1 - tpr
        
        eer_idx = np.nanargmin(np.abs(apcer_arr - bpcer_arr))
        eer = float((apcer_arr[eer_idx] + bpcer_arr[eer_idx]) / 2 * 100)
        
        def get_bpcer_at_apcer(target):
            valid_idx = np.where(apcer_arr <= target)[0]
            if len(valid_idx) > 0:
                return float(bpcer_arr[valid_idx[-1]] * 100)
            return float(bpcer_arr[0] * 100)
            
        results.update({
            'eer': eer,
            'bpcer10': get_bpcer_at_apcer(0.10),
            'bpcer20': get_bpcer_at_apcer(0.05),
            'bpcer100': get_bpcer_at_apcer(0.01),
        })
        
        # Save curve data for aggregation
        np.save(os.path.join(args.output, 'roc_data.npy'), {'fpr': fpr, 'tpr': tpr})
        print(f"Advanced metrics calculated -> EER: {eer:.2f}%, BPCER@10: {results['bpcer10']:.2f}%")
    except Exception as e:
        print(f"Failed to calculate advanced metrics: {e}")
    
    results_path = os.path.join(args.output, 'results_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {results_path}")
    print(f"ACE: {ace:.2f}%")

    # Clean up heavy model to save space
    '''if not args.keep_model:
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Deleted heavy model file to save space: {model_path}")'''
if __name__ == "__main__":
    main()