# test_ablation.py
import argparse
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import prepare_test_data
from model_ablation import LoRALayer

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained ablation model on the test set')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved .h5 model')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results_metrics.json')
    parser.add_argument('--config-note', type=str, default="", help='Optional label for this configuration')

    # --- Chemins du test set uniquement ---
    parser.add_argument('--real-dir-test', type=str, required=True, help='Path to testing Live/real images')
    parser.add_argument('--fake-dir-test', type=str, required=True, help='Path to testing Fake/spoof images')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    test_generator, all_labels_test = prepare_test_data(args.real_dir_test, args.fake_dir_test)

    custom_objects = {'LoRALayer': LoRALayer}
    model = load_model(args.model, custom_objects=custom_objects)

    # --- Test loss/accuracy (Keras), calcule une seule fois, jamais utilise pour l'entrainement ---
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = np.argmax(all_labels_test, axis=1)  # 1D labels

    # --- Encodage reel (cf. data_loader.py) : 0 = Live (real), 1 = Fake (spoof) ---
    live_indices = np.where(true_labels == 0)[0]
    spoof_indices = np.where(true_labels == 1)[0]

    # Standard threshold (0.5)
    apcer = np.mean(predicted_labels[spoof_indices] == 0) * 100  # spoof classe a tort comme live
    bpcer = np.mean(predicted_labels[live_indices] == 1) * 100   # live classe a tort comme spoof
    ace = (apcer + bpcer) / 2

    results = {
        'model_path': args.model,
        'config_note': args.config_note,
        'apcer': float(apcer),
        'bpcer': float(bpcer),
        'ace': float(ace),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
    }

    # Advanced Metrics & DET Curve Data
    try:
        from sklearn.metrics import roc_curve
        y_scores = predictions[:, 1]  # proba predite "fake" (classe 1)
        fpr, tpr, thresholds = roc_curve(true_labels, y_scores)

        # Classe positive = 1 (fake) dans roc_curve
        apcer_arr = 1 - tpr   # spoof non detecte (classe live)
        bpcer_arr = fpr       # live rejete a tort (classe fake)

        eer_idx = np.nanargmin(np.abs(apcer_arr - bpcer_arr))
        eer = float((apcer_arr[eer_idx] + bpcer_arr[eer_idx]) / 2 * 100)

        def get_bpcer_at_apcer(target):
            # apcer_arr est decroissant et bpcer_arr croissant avec l'index (cf. roc_curve).
            # On veut le BPCER minimal parmi les points qui respectent APCER <= target,
            # donc le PREMIER indice valide (et non le dernier, qui correspond au seuil
            # le plus bas -> fpr=1.0 -> bpcer=100% quel que soit target).
            valid_idx = np.where(apcer_arr <= target)[0]
            if len(valid_idx) > 0:
                return float(bpcer_arr[valid_idx[0]] * 100)
            return 100.0  # target APCER inatteignable sur cette courbe ROC

        results.update({
            'eer': eer,
            'bpcer10': get_bpcer_at_apcer(0.10),
            'bpcer20': get_bpcer_at_apcer(0.05),
            'bpcer100': get_bpcer_at_apcer(0.01),
        })

        np.save(os.path.join(args.output, 'roc_data.npy'), {'fpr': fpr, 'tpr': tpr})
        print(f"Advanced metrics calculated -> EER: {eer:.2f}%, BPCER@10: {results['bpcer10']:.2f}%")
    except Exception as e:
        print(f"Failed to calculate advanced metrics: {e}")

    results_path = os.path.join(args.output, 'results_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")
    print(f"ACE: {ace:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()