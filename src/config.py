from types import SimpleNamespace
import numpy as np
import torch

cfg = SimpleNamespace()

# paths
cfg.data_folder = ''
cfg.name = "singing_bird_detection"
cfg.data_dir = "./dataset/"
cfg.train_data_folder = cfg.data_dir + "train_audio/"
cfg.val_data_folder = cfg.data_dir + "train_audio/"
cfg.output_dir = "first_model"

# dataset
cfg.dataset = "base_ds"
cfg.min_rating = 0
cfg.val_df = None
cfg.batch_size_val = 1
cfg.train_aug = None
cfg.val_aug = None
cfg.test_augs = None
cfg.wav_len_val = 5  # seconds

# audio
cfg.window_size = 2048
cfg.hop_size = 512
cfg.sample_rate = 32000
cfg.fmin = 200
cfg.fmax = 16000
cfg.power = 2
cfg.mel_bins = 256
cfg.top_db = 80.0

# img model
cfg.backbone = "resnet18"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.train = True
cfg.val = False
cfg.in_chans = 1

cfg.alpha = 1
cfg.eval_epochs = 1
cfg.eval_train_epochs = 1
cfg.warmup = 0

cfg.mel_norm = False

cfg.label_smoothing = 0

cfg.remove_pretrained = []

# training
cfg.seed = 123
cfg.save_val_data = True

# ressources
cfg.mixed_precision = True
cfg.gpu = 0
cfg.num_workers = 4 # 18
cfg.drop_last = True

cfg.mixup2 = 0

cfg.label_smoothing = 0

cfg.mixup_2x = False


cfg.birds = np.array(['carduelis carduelis',
 'columba palumbus',
 'corvus corone/cornix',
 'erithacus rubecula',
 'fringilla coelebs',
 'parus major',
 'phylloscopus collybita',
 'turdus merula',
 'chloris chloris',
 'cyanistes caeruleus',
 'dryocopus martius',
 'passer montanus',
 'dendrocopos major',
 'phasianus colchicus',
 'picus canus or viridis',
 'sylvia atricapilla',
 'turdus philomelos',
 'aegithalos caudatus',
 'jynx torquilla',
 'luscinia megarhynchos',
 'phoenicurus ochruros',
 'cuculus canorus',
 'linaria cannabina',
 'motacilla alba',
 'emberiza citrinella',
 'picus canus',
 'lophophanes cristatus',
 'troglodytes troglodytes',
 'columba oenas',
 'prunella modularis',
 'regulus ignicapilla',
 'sitta europaea',
 'anthus trivialis',
 'sylvia communis',
 'turdus viscivorus',
 'poecile palustris',
 'strix aluco',
 'coccothraustes coccothraustes',
 'garrulus glandarius',
 'lanius excubitor',
 'bubo bubo',
 'corvus corax',
 'saxicola rubicola',
 'buteo buteo',
 'certhia brachydactyla',
 'sturnus vulgaris',
 'serinus serinus',
 'hirundo rustica',
 'hippolais icterina',
 'sylvia curruca',
 'emberiza calandra',
 'muscicapa striata',
 'coturnix coturnix',
 'periparus ater',
 'pica pica',
 'turdus pilaris',
 'corvus monedula',
 'loxia curvirrostra',
 'spinus spinus',
 'phoenicurus phoenicurus',
 'acrocephalus palustris',
 'merops apiaster'])


cfg.n_classes = len(cfg.birds)
# dataset
cfg.min_rating = 2.0

cfg.wav_crop_len = 30  # seconds

cfg.lr = 0.0001
cfg.epochs = 5
cfg.batch_size = 64
cfg.batch_size_val = 64
cfg.backbone = "resnet34"


cfg.save_val_data = True
cfg.mixed_precision = True

cfg.mixup = True
cfg.mix_beta = 1


cfg.train_df1 = "../input/birdclef-2022/train_metadata.csv"
cfg.train_df2 = "../input/birdclef-2022-df-train-with-durations/df-with-durations.csv"


cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg.tr_collate_fn = None
cfg.val_collate_fn = None
cfg.val = False

cfg.dev = True

cfg.model = "sing_bird"
