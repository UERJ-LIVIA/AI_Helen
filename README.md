# AI_Helen

# Packages imports
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from sklearn.preprocessing import StandardScaler #divisão de dados e pré-processamento
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import pysr
pysr.silence_julia_warning()
import json
import os
import dill as pickle
import time


# Preparing the data

''' class MyDataset():
 
  def __init__(self, data):

    data = data.values
 
    x = data[:, 0:-1]
    sc = StandardScaler() 
    
    y = data[:, -1]
 
    self.x_train=torch.tensor(x_sc, dtype=torch.float32) 
    #self.x_train=torch.tensor(x, dtype=torch.float32)
    self.x_train = self.x_train.to(device)
    
    self.y_train=torch.tensor(y, dtype=torch.float32)
    self.y_train = self.y_train.to(device)
    
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],  torch.unsqueeze(self.y_train[idx], dim=0) 
    
# Reading the data

def read_data(data_path, sep=','):
    print("Loading the data ...")
    file_extension = pathlib.Path(data_path).suffix

    if file_extension == '.txt':
        data = np.loadtxt(data_path)
        assert isinstance(data, np.ndarray), f"The input file must be a numpy array but is {type(data)}"
        col_names = ['X'+str(i+1) for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=col_names, dtype=np.float32)
    elif file_extension == '.csv':
        data = pd.read_csv(data_path, sep=sep, dtype=np.float32)
    else:
        print(f"the type of the data is {type(file_extension)} but a csv or saved numpy array is expected")
        raise TypeError()
    print("Reading the data is done!")
    return data  

# Neural Network constrution

def network(input_dimension,hidden_dimension, output_dimension):

  print("Constructing the neural network...")

  modules=[]
  modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
  modules.append(torch.nn.ReLU()) #função de ativação
  for i in range(len(hidden_dimension)-1):
    modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
    modules.append(torch.nn.ReLU())

  modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))

  net = torch.nn.Sequential(*modules) #incializar módulo interno

  if torch.cuda.is_available():
    net = net.to(device)
  
  print("Construction is done!")
  
  return net    

def train(net, train_loader, epochs=100, verbose=1):

  print("Training the neural network with the data...")
  n_show_epoch = int(epochs/10)

  loss_func = nn.MSELoss()


  optimizer = torch.optim.Adam(net.parameters()) #Gradient Descent
  scheduler = StepLR(optimizer, step_size=int(epochs/10), gamma=0.1, verbose=False)

# Training Neural Netowrk

  for e in range(epochs):
  
    running_loss= 0

    for features,labels in train_loader:
        
        output= net(features) 

        loss= loss_func(output, labels)
        optimizer.zero_grad() #zerar gradientens - não se quer acumulação
        loss.backward() #cálculo do gradiente de perda
        
        optimizer.step() #atualiza os parâmentros
        scheduler.step()
        

        running_loss += loss.item ()
    if verbose and e % n_show_epoch==0:
      print(f"epoch: {e}/{epochs}, loss: {loss.item()}")

  print(f"Training the neural network is done! The final loss is {loss.item()}")  
  
  return net
  
# Checking the simmetries  

def check_translational_symmetry_plus(model, data, min_error=0.05, shift=None, verbose=1):

    print("Checking the translational symmetry by plus...")
    data_translated = torch.from_numpy(data.values).to(device)
    #print(data_translated[:5, :])
    num_variables = data_translated.size()[1] - 1
    y = torch.unsqueeze(data_translated[:, -1], dim=1)

    if os.path.exists('columns.pkl'):
        columns = pickle.load("columns.pkl")
    else:
        columns = []
        


    with torch.no_grad():            
        
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                #if i<j:
                x = data_translated[:, 0:-1].clone()
                #print(data_translated[:5, :])


                if shift is None:
                    shift = 0.5 * min(torch.std(x[:,i]),torch.std(x[:,j]))

                x[:,i] += shift #(x[:,i] + shift).clone()
                x[:,j] += shift #(x[:,j] + shift).clone()
                errors = abs(y - model(x))
                error = torch.median(errors)
                if verbose:
                    print(f"Columns {(i+1,j+1)} -> error {error}")
                if error < min_error:
                    #columns[(i+1,j+1)] = float(error)
                    columns.append([i+1, j+1])
    with open('columns.pkl', 'wb') as f:
        pickle.dump(columns, 'columns.pkl')
        
    print("Checking the translational symmetry is done!")
    
    #return columns
    
# Saving

def save_dict(dict_path, obj):
    with open(dict_path, 'w') as convert_file:
        convert_file.write(json.dumps(obj))

def read_dict(dict_path):
    with open(dict_path) as json_file:
        d = json.load(json_file)
    return d
    
# Applying the translational simetry      

def apply_translational_symmetry(data, columns):
    print(f"Reducing the number of indepent variables by {len(columns)}")


    if os.path.exists('variable_change.json'):
        variable_change = read_dict('variable_change.json')
    else:
        variable_change = dict()

    data_new = data.iloc[:, 0:-1].copy()
    for i, j in columns:
        
        if data_new.columns[-1][0] == 'X':
            col_num = 1
        else:
            col_num = int(data_new.columns[-1][1:]) + 1
        data_new['u'+str(col_num)] = data_new.iloc[:, i-1] - data_new.iloc[:, j-1]
        variable_change['u'+str(col_num)] = data_new.columns[i-1] + '-' + data_new.columns[j-1]



    #drop_cols = list(set([i for l in columns.keys() for i in l])) # the list of the columns to be deleted
    #drop_cols = [i-1 for i in drop_cols] #correcting the index of the columns to be deleted
    drop_cols = [i-1 for sublist in columns for i in sublist]

    data_new = data_new.drop(labels=data.columns[drop_cols], axis=1)
    data_new = pd.concat([data_new, data.iloc[:, -1]], axis=1)
    
    data_new.to_csv('data_new.csv', index=False)
    save_dict('variable_change.json', variable_change)

    return data_new, variable_change    
 
 # Including genetic alghoritm
 
 def SR_GA(path_data, number_of_samples=500, iterations=10):
    data = pd.read_csv(path_data)
    indices = np.random.choice(data.shape[0], number_of_samples, replace=False)
    X = data.iloc[indices, 0:-1]  #separando variáveis de entrada
    y = data.iloc[indices, -1].values

    sr_model = pysr.PySRRegressor(
    niterations=iterations,
    #populations=10,
    binary_operators=["+", "*", "-", "/", "pow"],
    #batching=True,
    #batchSize = 128,
    #procs = 20,
    #multithreading = True,
    unary_operators=[
        "cos",
        "exp",
        "sin",  # Pre-defined library of operators (see docs)
        "inv(x) = 1/x",  # Define your own operator! (Julia syntax)
                    ],
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
    )

    sr_model.fit(X, y)

 
    sr_model.equations.to_csv('results.csv', index=False)
    #sr_model.raw_julia_state = None

    #with open('sr_model.pkl', 'wb') as sr_model_file:
    #    pickle.dump(sr_model, sr_model_file)



    return sr_model

# Showing results
def show_results(sr_model=None, all_results=False):
    """shows the result of symbolic regression using the PySR

    Args:
        sr_model (PySR model, optional): a fitted PySR model on data. Defaults to None.
        best (bool, optional): True means printing the best model. False means printing all the models. Defaults to True.
    """

    time.sleep(2)
    
    variable_change = read_dict('variable_change.json')
    # the below part for loading isn't working right now
    if sr_model is None:
        with open('sr_model.pk', 'rb') as sr_model_file:
            sr_model = pickle.load(sr_model_file)

    if all_results is False:
        eq = sr_model.sympy()
        eq = eq.subs(variable_change).simplify()
        print("--------------------------\n")
        print("The Final Result is:\n")
        print(eq)
        #return eq
    else:
        print("--------------------------\n")
        print("The Final Result is:\n")
        print(sr_model.equations.replace(variable_change, regex=True).loc[:, ['loss', 'equation']])
        #return sr_model.equations.replace(variable_change, regex=True).loc[:, ['loss', 'equation']]
        
# Set of all functions

def run_regression(data_path, 
                    epochs=100,
                    iterations=10,
                    hidden_layers = [200, 200, 100],
                    all_results=False,
                    min_error=0.01,
                    number_of_samples=500,
                    batch_size = 1028,
                    verbose=1 ):
    """run a neural_genetic symbolic regression on data

    Args:
        data_path (_type_): _description_
        epochs (int, optional): _description_. Defaults to 100.
        iterations (int, optional): _description_. Defaults to 10.
        hidden_layers (list, optional): _description_. Defaults to [200, 200, 100].
        best_result (bool, optional): _description_. Defaults to True.
        min_error (float, optional): _description_. Defaults to 0.05.
        number_of_samples (int, optional): _description_. Defaults to 500.
        batch_size (int, optional): _description_. Defaults to 1028.

    Returns:
        _type_: A fitted model of PySR type
    """
                    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    data = read_data(data_path)
    myDs=MyDataset(data)
    train_loader=DataLoader(myDs, batch_size=batch_size, shuffle=True)
    dim_features = data.shape[1] - 1
    net= network(dim_features, hidden_layers, 1)
    model = train(net, train_loader, epochs=epochs, verbose=verbose)
    columns = check_translational_symmetry_minus(model, data, min_error=0.05, verbose=verbose)
    data_new, variable_change = apply_translational_symmetry(data, columns)    
    sr_model = SR_GA('data_new.csv', iterations=iterations, number_of_samples=number_of_samples)
    show_results(sr_model)
    
    return sr_model
