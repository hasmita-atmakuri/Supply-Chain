# Dataset Requirements

## Expected Dataset Format

Your dataset should be a **CSV file** with the following requirements:

### Required Format:
1. **File Type**: CSV (Comma-Separated Values)
2. **Encoding**: The code uses 'latin1' encoding, but UTF-8 should also work
3. **Structure**: 
   - Multiple feature columns (input variables)
   - One target/label column (output variable to predict)

### Target Column (Label Column):
The code will automatically detect the target column by looking for these names (in order):
- `Order Status` (preferred)
- `order_status`
- `OrderStatus`
- `target`
- `Target`
- `label`
- `Label`
- `class`
- `Class`

**If none of these are found**, the code will use the **last column** as the target.

### Data Types:
- **Features**: Can be numeric or categorical (text)
- **Target**: Must be categorical/classification labels (discrete classes)
- The code automatically:
  - Encodes categorical columns to numbers
  - Handles missing values
  - Applies SMOTE oversampling for class balancing

### Example Dataset Structure:
```
Feature1, Feature2, Feature3, Order Status
value1,   value2,   value3,   Delivered
value4,   value5,   value6,   Pending
value7,   value8,   value9,   Cancelled
...
```

### Important Notes:
1. **Classification Problem**: Your dataset should be for classification (predicting categories/classes), not regression (predicting numbers)
2. **At least 2 classes**: The target column must have at least 2 different class labels
3. **Multiple features**: You need at least one feature column (besides the target)
4. **No header issues**: Make sure the CSV has proper column headers

### When you run the application:
- After uploading, check the console output to see which column is being used as the target
- The application will show a warning if it can't find the expected column name
- All available columns will be listed in the console








