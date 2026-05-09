def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    for i,value in enumerate(values):
        if value == None:
            right = i+1
            while values[right] is None:
                right+=1
            values[i] = values[i-1] + (i - (i-1))/ (right - (i-1))*(values[right] - values[i-1])


    return values