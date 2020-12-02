
from torchvision import transforms

from torchvision.transforms.transforms import *
from pre_process import image_train, image_test

config= {
            'office31': {
               'image_size':256,
               'task_list':['amazon','dslr','webcam'],
               'num_classes':31,
                'train_transform': image_train(resize_size=256, crop_size=224),
                'test_transform': image_test(resize_size=256, crop_size=224),
                'need_balance':True,
                'num_classes':31,
             'batch_size':16,
             'test_batch_size':16,
             'data_list': {'train':
             {'amazon':'./data/data_list/office31/amazon_reorgnized.txt',
             'dslr':'./data/data_list/office31/dslr_reorgnized.txt',
             'webcam':'./data/data_list/office31/webcam_reorgnized.txt'},
             'test':
             {'amazon':'./data/data_list/office31/amazon_reorgnized.txt',
             'dslr':'./data/data_list/office31/dslr_reorgnized.txt',
             'webcam':'./data/data_list/office31/webcam_reorgnized.txt'}}
             },
             
     'pacs':{'image_size':225,
    'task_list':["art_painting", "cartoon", "photo", "sketch"],
    'batch_size': 64,
    'test_batch_size': 16,
     'train_transform':          
          transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
         'test_transform':
         transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
          'need_balance':True,
          'num_classes':7,
    'data_list':{'train':{'art_painting':'./data/data_list/pacs/art_painting_train_kfold.txt',
                             'photo': './data/data_list/pacs/photo_train_kfold.txt',
                             'sketch':'./data/data_list/pacs/sketch_train_kfold.txt',
                             'cartoon':'./data/data_list/pacs/cartoon_train_kfold.txt'
                            },
                'test':{'art_painting':'./data/data_list/pacs/art_painting_test_kfold.txt',
                             'photo': './data/data_list/pacs/photo_test_kfold.txt',
                             'sketch':'./data/data_list/pacs/sketch_test_kfold.txt',
                             'cartoon':'./data/data_list/pacs/cartoon_test_kfold.txt'
                            }
                            }
} ,
    
    'office_home':{'image_size':256,
         'task_list':['Art', 'Clipart', 'Product', 'Real_World'],
        'train_transform': image_train(resize_size=256, crop_size=224),
         'test_transform':
          image_test(resize_size=256, crop_size=224),
    
          'need_balance':True,
          'num_classes':65,
             'batch_size':24,
             'test_batch_size':32,
            
        'data_list':{'train':{
    'Art':{'5':'data/data_list/office_home/Art/train_5.txt',
           '10':'data/data_list/office_home/Art/train_10.txt' ,
          '20': 'data/data_list/office_home/Art/train_20.txt',
          'all': 'data/data_list/office_home/Art/train.txt'},
    'Clipart':{'5':'data/data_list/office_home/Clipart/train_5.txt',
        '10':'data/data_list/office_home/Clipart/train_10.txt',
        '20':'data/data_list/office_home/Clipart/train_20.txt',
        'all':'data/data_list/office_home/Clipart/train.txt'},
    'Product':{
        '5':'data/data_list/office_home/Product/train_5.txt',
        '10':'data/data_list/office_home/Product/train_10.txt',
        '20':'data/data_list/office_home/Product/train_20.txt',
        'all':'data/data_list/office_home/Product/train.txt'
    },
    'Real_World':{
        '5':'data/data_list/office_home/Real_World/train_5.txt',
        '10':'data/data_list/office_home/Real_World/train_10.txt',
        '20':'data/data_list/office_home/Real_World/train_20.txt',
        'all':'data/data_list/office_home/Real_World/train.txt',
    }},
        'test':{    
    'Art':{'5':'data/data_list/office_home/Art/test_5.txt' ,
        '10':'data/data_list/office_home/Art/test_10.txt' ,
          '20': 'data/data_list/office_home/Art/test_20.txt',
          'all': 'data/data_list/office_home/Art/test.txt'},
    'Clipart':{
        '5':'data/data_list/office_home/Clipart/test_5.txt',
        '10':'data/data_list/office_home/Clipart/test_10.txt',
        '20':'data/data_list/office_home/Clipart/test_20.txt',
        'all':'data/data_list/office_home/Clipart/test.txt'},
    'Product':{
        '5':'data/data_list/office_home/Product/test_5.txt',
        '10':'data/data_list/office_home/Product/test_10.txt',
        '20':'data/data_list/office_home/Product/test_20.txt',
        'all':'data/data_list/office_home/Product/test.txt'
    },
    'Real_World':{
        '5':'data/data_list/office_home/Real_World/test_5.txt',
        '10':'data/data_list/office_home/Real_World/test_10.txt',
        '20':'data/data_list/office_home/Real_World/test_20.txt',
        'all':'data/data_list/office_home/Real_World/test.txt',
        }
            }}},

    'office_caltech': {'image_size':256,
            'task_list':['amazon','caltech','dslr','webcam'],
                    'train_transform': image_train(resize_size=256, crop_size=224),
         'test_transform':
          image_test(resize_size=256, crop_size=224),
          'need_balance':True,
          'num_classes':10,
             'batch_size':16,
             'test_batch_size':16,
         
        'data_list':{'train':{
    'amazon':{'5':'data/data_list/office_caltech/amazon/train_5.txt',
           '10':'data/data_list/office_caltech/amazon/train_10.txt' ,
          '20': 'data/data_list/office_caltech/amazon/train_20.txt',
          'all': 'data/data_list/office_caltech/amazon/train.txt'},
    'caltech':{'5':'data/data_list/office_caltech/caltech/train_5.txt',
        '10':'data/data_list/office_caltech/caltech/train_10.txt',
        '20':'data/data_list/office_caltech/caltech/train_20.txt',
        'all':'data/data_list/office_caltech/caltech/train.txt'},
    'webcam':{
        '5':'data/data_list/office_caltech/webcam/train_5.txt',
        '10':'data/data_list/office_caltech/webcam/train_10.txt',
        '20':'data/data_list/office_caltech/webcam/train_20.txt',
        'all':'data/data_list/office_caltech/webcam/train.txt'
    },
    'dslr':{
        '5':'data/data_list/office_caltech/dslr/train_5.txt',
        '10':'data/data_list/office_caltech/dslr/train_10.txt',
        '20':'data/data_list/office_caltech/dslr/train_20.txt',
        'all':'data/data_list/office_caltech/dslr/train.txt',
    }},
        'test':{    
    'amazon':{'5':'data/data_list/office_caltech/amazon/test_5.txt' ,
        '10':'data/data_list/office_caltech/amazon/test_10.txt' ,
          '20': 'data/data_list/office_caltech/amazon/test_20.txt',
          'all': 'data/data_list/office_caltech/amazon/test.txt'},
    'caltech':{
        '5':'data/data_list/office_caltech/caltech/test_5.txt',
        '10':'data/data_list/office_caltech/caltech/test_10.txt',
        '20':'data/data_list/office_caltech/caltech/test_20.txt',
        'all':'data/data_list/office_caltech/caltech/test.txt'},
    'webcam':{
        '5':'data/data_list/office_caltech/webcam/test_5.txt',
        '10':'data/data_list/office_caltech/webcam/test_10.txt',
        '20':'data/data_list/office_caltech/webcam/test_20.txt',
        'all':'data/data_list/office_caltech/webcam/test.txt'
    },
    'dslr':{
        '5':'data/data_list/office_caltech/dslr/test_5.txt',
        '10':'data/data_list/office_caltech/dslr/test_10.txt',
        '20':'data/data_list/office_caltech/dslr/test_20.txt',
        'all':'data/data_list/office_caltech/dslr/test.txt',
        }
            }}}
            }
