import math
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from model import EnsembleModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def np2img(array):
    array = np.array(array)
    if len(array.shape) > 3:
        PIL_image = [Image.fromarray(image.astype('uint8'), 'RGB') for image in array]
    else:
        PIL_image = Image.fromarray(array.astype('uint8'), 'RGB')
    return PIL_image

def transform_img(img):
    img = np.array(img)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),   
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),               
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
    
    if len(img.shape) == 4:
        return np.array([transform(np2img(img[i])).permute([1,2,0]).numpy()*255 for i in range(img.shape[0])])
            
    elif len(img.shape) == 3:
        img = np2img(img)
        return transform(img).permute([1,2,0]).numpy()*255
    else:
        assert 0
        

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, transform=lambda x: x):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.transform = transform
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        batch = [self.dataset[int(i)].values() for i in batch_indices]

        self.current_idx = end_idx

        imgs, labels, sources = zip(*batch)
        labels = torch.tensor(labels)
        imgs = torch.tensor(self.transform(imgs), dtype=torch.float).permute(0,3,1,2)
        return imgs, labels, sources
    def __len__(self):
        return len(self.dataset)
    
    
    
    

def print_accuracy(model, dataloader, is_YOLO = False):
    score = 0
    total = 0
    wrong_probs = []
    correct_probs = []
    if not is_YOLO:
        model.eval()
    with tqdm(total = len(dataloader)//dataloader.batch_size) as pbar:
        with torch.no_grad():
            for images, labels, srcs in dataloader:
                images, labels = images.to(device), labels.to(device)
                if is_YOLO:
                    #assert onehotdecoder is not None, "Onehotdecoder is required for YOLO model"
                    pred = model.predict(srcs, verbose=False)
                    predicted = torch.tensor(list(map(lambda x: x.probs.top1, pred)))
                    prob = list(map(lambda x: x.probs.top1conf , pred))
                else:
                    outputs = model(images)
                    outputs = F.softmax(outputs, 1)
                    prob, predicted = torch.max(outputs, 1)
                    prob = prob.tolist()
                total += labels.size(0)
                correct = (predicted == labels)
                for i, is_correct in enumerate(correct):
                    if not is_correct:
                        wrong_probs.append(prob[i])
                    else: 
                        correct_probs.append(prob[i])
                score += correct.sum().item()
                pbar.update(1)

    acc = 100 * score / total
    print('Accuracy of the network on the %d test images: %d %%' % (total,
        acc))
    wrong_probs, correct_probs = torch.tensor(wrong_probs), torch.tensor(correct_probs)
    print(wrong_probs.mean())
    print(correct_probs.mean())

    idx = np.arange(1, 3)
    labels = ['wrong_probs', 'correct_probs']
    plt.boxplot([wrong_probs, correct_probs])
    plt.xticks(idx, labels)
    return acc
    
    
    
def train(model, dataset, valid_dataset, lr, batch_size, epochs, print_iter_num, load_path = None, save_path = None):
    save_loss = []
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    loss_accumulated = 0
    loaded_epoch = 0
    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    if load_path is not None:
        state = torch.load(load_path)
        model.load_state_dict(state["model"])
        model = model.to(device)
        optimizer.load_state_dict(state["optimizer"])
        loaded_epoch = state["epoch"]
        save_loss = state["save_loss"]
        best_acc = state["best_acc"]

    for epoch in np.array(list(range(epochs - loaded_epoch))) + loaded_epoch:
        print(f'epoch {epoch+1}')
        loss_accumulated = 0
        #acc = print_accuracy(model, valid_dataloader)
        
        model.train()  
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, transform=transform_img)
        with tqdm(total = len(dataloader)//batch_size) as pbar:
            for i, (imgs, labels, srcs) in enumerate(dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                with torch.no_grad():
                    score = F.softmax(outputs, 1)
                    #ans = score[torch.arange(labels.shape[0]), labels]
                    
                    
                
                loss = criterion(outputs, labels)

                #loss = criterion(outputs, labels)
                with torch.no_grad():
                    loss_accumulated += loss
                    if (i+1) % print_iter_num == 0:
                        print('loss: ',loss_accumulated/print_iter_num)
                        save_loss.append((loss_accumulated/print_iter_num).cpu().detach().numpy())
                        loss_accumulated = 0
                loss.backward()
                optimizer.step()
                pbar.update(1)
                
        acc = print_accuracy(model, valid_dataloader)
                
        if best_acc < acc:
            best_acc = acc
            print('----best_model updated----')
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "save_loss": save_loss,
                'best_acc' : best_acc,
                #"best_valid_loss": best_valid_loss,
            }
            torch.save(state, save_path)
            
def ads(train_img_list):
    
    from glob import glob
 
    # 이미지들의 주소 리스트로 만들어줌
    train_img_list = glob('./dataset/train/images/*.jpg')
    valid_img_list = glob('./dataset/valid/images/*.jpg')
    
    # 리스트를 txt파일로 생성
    with open('./dataset/train.txt', 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')
        
    with open('./dataset/valid.txt', 'w') as f:
        f.write('\n'.join(valid_img_list) + '\n')            
        
        
        
        
        
        
            
def train_coteaching(teacher, student, dataset, test_dataset, batch_size, epochs, lr, print_iter_num = 10, save_path = './model.pt', load_path = None
):
    
    
    loaded_epoch = 0
    best_acc = 0
    save_loss = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    
    
    if load_path is not None:
        state = torch.load(load_path)
        #teacher.load_state_dict(state["teacher"])
        student.load_state_dict(state["student"])
        student = student.to(device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        loaded_epoch = state["epoch"]
        save_loss = state["save_loss"]
        best_acc = state["best_acc"]
    
    
    
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    student = student.to(device)
    
    for epoch in np.array(list(range(epochs - loaded_epoch))) + loaded_epoch:
        print(f'epoch {epoch+1}')
        student.train()
        #splited_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, transform=lambda x: transform_img(x))
        loss_accumulated = 0
        with tqdm(total = len(dataloader)//batch_size) as pbar:
            for i, (imgs, labels, srcs) in enumerate(dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                
                
                pred = teacher.predict(srcs, verbose=False)
                score1 = torch.stack(list(map(lambda x: x.probs.data, pred)), 0).to(device)
                #prob = list(map(lambda x: x.probs.top1conf , pred))
                
                optimizer.zero_grad()
                outputs = student(imgs)
                score2 = F.softmax(outputs, 1)
                ans1 = score1[torch.arange(labels.shape[0]), labels]
                ans2 = score2[torch.arange(labels.shape[0]), labels]
                _, topk_idx = (ans1+ans2).topk(labels.shape[0]* 9 // 10)
                
                
                loss = criterion(outputs[topk_idx], labels[topk_idx])
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    loss_accumulated += loss
                    if (i+1) % print_iter_num == 0:
                        print('loss: ',loss_accumulated/print_iter_num)
                        save_loss.append((loss_accumulated/print_iter_num).cpu().detach().numpy())
                        loss_accumulated = 0
                
                pbar.update(1)
            
    
            acc = print_accuracy(student, valid_dataloader)
            
            if best_acc < acc:
                best_acc = acc
                print('----best_model updated----')
                state = {
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "save_loss" : save_loss,
                    'best_acc' : best_acc,
                    #"best_valid_loss": best_valid_loss,
                }
                torch.save(state, save_path)
                
            #print_accuracy(EnsembleModel(teacher, student), valid_dataloader)  
             
            
            scheduler.step()





