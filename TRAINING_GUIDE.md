# 🎯 CVRP Model Training & Demonstration Guide

## 📋 **Complete Training & Testing Workflow**

### **Phase 1: Quick Demo (5 minutes)**

```bash
# Run the demo to see everything working
python demo.py
```

**What this shows:**

- ✅ Data generation with ground truth solutions
- ✅ Model architecture working (encoder + decoder)
- ✅ Training pipeline functional
- ✅ Inference capability
- ✅ Debug logging throughout

### **Phase 2: Full Training (30-60 minutes)**

```bash
# Train a proper model with 500 instances over 30 epochs
python train_cvrp.py
```

**What this does:**

- Generates 500 training + 50 validation instances
- Trains for 30 epochs with batch size 16
- Saves trained model as `trained_cvrp_model.pt`
- Creates training progress plot
- Tests on validation instances

### **Phase 3: Benchmark Testing**

```bash
# Test your trained model on standard benchmarks, and save the visualizations
python test_cvrp_benchmarks.py

# Test without saving visualizations
python test_cvrp_benchmarks.py --no-plots

# Or test on a sample instance with detailed output
python test_cvrp_benchmarks.py --sample
```

---

## 🚀 **Detailed Training Instructions**

### **1. Understanding Your Model Architecture**

Your CVRP neural network has these components:

```python
# Input: [depot_coordinates, customer_coordinates, demands, capacity]
# ↓
# Embedding Layer: Convert coordinates + demands to 128-dim vectors
# ↓  
# Encoder: 3-layer transformer with 8 attention heads
# ↓
# Decoder: Attention-based next-node selection with capacity masking
# ↓
# Output: Next customer to visit
```

### **2. Training Process Explained**

**Supervised Learning Approach:**

1. **Generate Random Instances**: Create CVRP problems with random coordinates/demands
2. **Solve with Traditional Algorithm**: Use nearest neighbor + 2-opt for ground truth
3. **Train Neural Network**: Learn to predict the next node in optimal routes
4. **Validate**: Test on unseen instances and compare distances

### **3. Training Parameters (Configurable)**

```python
# Model Architecture
model_params = {
    'embedding_dim': 128,        # Higher = more capacity, slower training
    'encoder_layer_num': 3,      # More layers = more complex patterns
    'head_num': 8,              # Multi-head attention
    'debug_mode': False         # Set True for detailed logging
}

# Training Settings  
training_params = {
    'learning_rate': 1e-4,      # Lower = more stable, slower
    'epochs': 30,               # More epochs = better performance
    'batch_size': 16,           # Higher = faster, needs more memory
    'n_train': 500,             # More data = better generalization
    'n_customers': 20           # Problem size
}
```

### **4. Expected Training Timeline**

| Stage | Time | Output |
|-------|------|--------|
| Data Generation | 10-20 min | 500 training instances with optimal solutions |
| Model Training | 20-30 min | 30 epochs, loss decreasing |
| Validation | 2-5 min | Performance on unseen instances |
| **Total** | **30-60 min** | **Trained model ready** |

---

## 📊 **Performance Expectations**

### **Training Progress Indicators**

**Good Training Signs:**

- ✅ Loss decreasing steadily
- ✅ Validation distance improving
- ✅ Gap to optimal solutions < 30% after training

**Warning Signs:**

- ⚠️ Loss not decreasing after 10 epochs
- ⚠️ Validation distance getting worse
- ⚠️ Gap to optimal > 50%

### **Performance Benchmarks**

After proper training, expect these results:

| Problem Size | Expected Gap | Confidence |
|--------------|--------------|------------|
| 20 customers | 10-20% | High |
| 50 customers | 15-25% | Medium |
| 100+ customers | 20-35% | Lower |

---

## 🛠️ **Training Options & Customization**

### **Option 1: Quick Training (Development)**

```python
# In train_cvrp.py, modify these lines:
trainer.generate_training_data(
    n_train=100,        # Smaller dataset
    n_val=20,           # Quick validation
    n_customers=15      # Smaller problems
)

train_losses, val_distances = trainer.train_supervised(
    epochs=10,          # Fewer epochs
    batch_size=8        # Smaller batches
)
```

### **Option 2: Production Training (Best Results)**

```python
# In train_cvrp.py, modify these lines:
trainer.generate_training_data(
    n_train=2000,       # Large dataset
    n_val=200,          # Thorough validation
    n_customers=50      # Larger problems
)

train_losses, val_distances = trainer.train_supervised(
    epochs=100,         # More epochs
    batch_size=32       # Larger batches (if GPU memory allows)
)
```

### **Option 3: Debug Training (Understanding)**

```python
# In train_cvrp.py, set debug mode:
model_params = {
    # ... other params ...
    'debug_mode': True      # Detailed logging
}
```

---

## 🧪 **Testing & Validation**

### **Method 1: Sample Instance Testing**

```bash
python test_cvrp_benchmarks.py --sample
```

**Shows:**

- Generated instance details
- Step-by-step neural network decisions (if debug=True)
- Comparison with optimal solution
- Performance assessment

### **Method 2: Benchmark Dataset Testing**

1. **Download benchmark instances:**

   ```bash
   # Create benchmark directory
   mkdir benchmark_data
   
   # Download from: http://vrp.atd-lab.inf.puc-rio.br/index.php/en/
   # Suggested files: A-n32-k5.vrp, A-n45-k6.vrp, B-n31-k5.vrp
   ```

2. **Run benchmark tests:**

   ```bash
   python test_cvrp_benchmarks.py
   ```

### **Method 3: Custom Instance Testing**

```python
# Create your own test instance
from cvrp_data_generator import CVRPDataGenerator

generator = CVRPDataGenerator()
test_instance = generator.generate_dataset(
    n_instances=1,
    n_customers=25,
    capacity_range=(100, 150),
    demand_range=(10, 25)
)[0]

# Test with your trained model
predicted_distance = trainer._solve_instance(test_instance)
```

---

## 📈 **Improving Performance**

### **If Results Are Poor (Gap > 40%)**

1. **More Training Data:**

   ```python
   n_train=1000  # or higher
   ```

2. **Longer Training:**

   ```python
   epochs=50  # or higher
   ```

3. **Better Ground Truth:**
   - The traditional solver (nearest neighbor + 2-opt) provides ~10% gap solutions
   - Consider implementing more advanced heuristics

4. **Model Architecture:**

   ```python
   embedding_dim=256,      # Larger model
   encoder_layer_num=6,    # More layers
   ```

### **If Training Is Too Slow**

1. **Smaller Problems:**

   ```python
   n_customers=15
   ```

2. **Smaller Datasets:**

   ```python
   n_train=200
   ```

3. **CPU Training:**

   ```python
   use_cuda=False
   ```

---

## 🎯 **Success Criteria**

### **After Quick Demo:**

- ✅ Pipeline runs without errors
- ✅ Model produces feasible solutions
- ✅ Debug output shows attention working

### **After Full Training:**

- ✅ Training loss decreases to < 1.0
- ✅ Validation gap < 25% on average
- ✅ Model saves and loads correctly

### **After Benchmark Testing:**

- ✅ Solutions are feasible (all customers visited)
- ✅ Gap to known optimal < 30%
- ✅ Consistent performance across instances

---

## 🚀 **Ready to Start!**

Your complete CVRP neural network solution is ready. Follow these steps:

1. **Start with demo:** `python demo.py`
2. **Train full model:** `python train_cvrp.py`
3. **Test performance:** `python test_cvrp_benchmarks.py --sample`
4. **Benchmark validation:** Download .vrp files and run `python test_cvrp_benchmarks.py`
