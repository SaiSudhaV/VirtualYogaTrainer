#!/usr/bin/env python3
"""
Yoga Pose Detection - Dataset Preparation & Model Training Pipeline
"""

import os
import time
from data_preprocessor import YogaDataPreprocessor
from model_trainer import YogaModelTrainer

def main():
    print("🧘 YOGA POSE DETECTION - TRAINING PIPELINE 🧘")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Initialize Data Preprocessor
    print("\n📊 STEP 1: DATASET PREPARATION & PREPROCESSING")
    print("-" * 40)
    
    preprocessor = YogaDataPreprocessor()
    
    # Step 2: Load Dataset
    print("Loading dataset from images...")
    try:
        X, y = preprocessor.load_dataset_from_images()
        
        if len(X) == 0:
            print("❌ No valid data found. Please check your dataset.")
            return
        
        print(f"✅ Successfully loaded {len(X)} samples with {X.shape[1]} features")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Step 3: Preprocess Data
    print("\nPreprocessing data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(X, y)
        
        print(f"✅ Data preprocessing completed:")
        print(f"  • Training set: {X_train.shape[0]} samples")
        print(f"  • Validation set: {X_val.shape[0]} samples") 
        print(f"  • Test set: {X_test.shape[0]} samples")
        print(f"  • Feature dimensions: {X_train.shape[1]}")
        
        # Save preprocessor
        preprocessor.save_preprocessor()
        print("✅ Preprocessor saved")
        
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        return
    
    # Step 4: Model Training & Tuning
    print(f"\n🤖 STEP 2: MODEL TRAINING & TUNING")
    print("-" * 40)
    
    try:
        trainer = YogaModelTrainer(preprocessor.pose_names)
        results = trainer.train_and_tune_models(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return
    
    # Step 5: Evaluate Best Model
    print(f"\n📊 STEP 3: MODEL EVALUATION")
    print("-" * 40)
    
    try:
        trainer.evaluate_best_model(X_test, y_test)
        
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        return
    
    # Step 6: Save Best Model
    print(f"\n💾 STEP 4: SAVING MODEL")
    print("-" * 40)
    
    try:
        trainer.save_model_info()
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Final Summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 50)
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"🏆 Best model: {trainer.best_model_name}")
    print(f"🎯 Best validation accuracy: {trainer.best_score:.3f}")
    print(f"📁 Files generated:")
    print(f"  • best_model_info.json (model information)")
    print(f"  • preprocessor.json (data preprocessor)")

if __name__ == "__main__":
    main()