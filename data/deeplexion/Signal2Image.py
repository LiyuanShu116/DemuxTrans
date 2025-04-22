import torch
from pyts.image import MarkovTransitionField, GramianAngularField, RecurrencePlot

image_size=224

# X = pd.read_feather('X.feather').to_numpy()
def pyts_transform(transform,data):
    try:
        X_transform = transform.fit_transform(data)
        return(X_transform)
    except Exception as e:
        print('Error:', str(e))
        return([])

def gasf_transform(data, image_size=224):
    # GAF transformation
    transform = GramianAngularField(image_size, method='summation')
    return(pyts_transform(transform,data))


def signal2image(data):     #(n, 700)
    final_signals = []

    for signal in data:
        transformed_squiggle = gasf_transform(signal.reshape(1, -1), image_size=image_size)
        transformed_squiggle_tensor = torch.from_numpy(transformed_squiggle).unsqueeze(0)  # (1, 1, 224, 224)
        final_signals.append(transformed_squiggle_tensor)

    final_signals = torch.cat(final_signals, dim=0)  # (n, 1, 224, 224)

    return final_signals

# imgs = signal2image(X)
#
# with h5py.File('X_Image.h5', 'w') as h5f:
#     h5f.create_dataset('dataset', data=imgs.numpy())

def signal2image(data):     #(n, 700)
    final_signals = []

    for signal in data:
        transformed_squiggle = gasf_transform(signal.reshape(1, -1), image_size=image_size)
        transformed_squiggle_tensor = torch.from_numpy(transformed_squiggle).unsqueeze(0)  # (1, 1, 224, 224)
        final_signals.append(transformed_squiggle_tensor)

    final_signals = torch.cat(final_signals, dim=0)  # (n, 1, 224, 224)

    return final_signals