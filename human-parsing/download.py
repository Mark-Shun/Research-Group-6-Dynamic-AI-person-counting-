import gdown

dataset = 'data'         #select from ['lip', 'atr', 'pascal']

if dataset == 'lip':
    url = 'https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH'
elif dataset == 'atr':
    url = 'https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP'
elif dataset == 'pascal':
    url = 'https://drive.google.com/uc?id=1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE'
elif dataset == 'data':
    url = 'https://drive.google.com/file/d/1eT7cksAdIUgNa_M-qtsK02waKqDsCrmZ/view?usp=drive_link'

if dataset == 'data':
    output = 'data.zip'
else:
    output = 'checkpoints/final.pth'
gdown.download(url, output, quiet=False, fuzzy=True)
