# Progetto ML: Predizione della Struttura Secondaria delle Proteine
Il progetto è realizzato come elaborato finale del corso di Machine Learning presso l'Università degli Studi di Firenze, anno accademico 2022-2023. L'obiettivo è sviluppare architetture per la classificazione multiclasse delle sequenze proteiche, sfruttando le conoscenze acquisite durante la frequentazione del corso, eseguendo inoltre un'accurata fase di testing. La predizione è realizzata su 8 classi, sviluppando due differenti tipologie di reti neurali:
- deep neural network with self-attention mechanism
- convolutional recurrent neural network

Le reti sono state addrestrate accedendo da remoto alle risorse fornite dalla piattaforma Google Colab.

![alt text](https://github.com/Tiezzi96/progettoML/blob/main/PSL_protein_structure.it_IT.x512.png?raw=true)

## Dataset
Il dataset utilizzato è quello a cui fa riferimento l'articolo di Zhou & Troyanskaya ed è scaricabile al seguente link: <a href="https://www.princeton.edu/~jzthree/datasets/ICML2014/" target="_blank">ICML2014</a>

Durante la realizzazione dell'elaborato sono stati impiegati i seguenti dataset:
- *CullPDB+5916filtered.npy* per l'addestramento
- *CB513+split1.npy* per la fase di testing 


# Bibliografia
- <a href="https://academic.oup.com/bioinformatics/article/36/17/4599/5841663" target="_blank">SAINT Article</a>
- <a href="https://www.ijcai.org/Abstract/16/364" target="_blank">Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks</a>
