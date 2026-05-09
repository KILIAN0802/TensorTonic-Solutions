import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    L = []
    # 1. Chuyển sang numpy array và lấy số lượng class
    
    num_classes = len(predictions)
    
    # 2. Vòng lặp tính q và loss từng phần tử
    for i, p in enumerate(predictions):
        if i == target:
            q = (1 - epsilon) + (epsilon / num_classes)
        else:
            q = epsilon / num_classes
        
        # Thêm giá trị vào list L (nhớ lùi vào đúng bằng hàng trên)
        L.append(-q * np.log(p + 1e-12))
        
    # 3. Trả về tổng (dòng này phải thẳng hàng với L = [])
    return float(sum(L))