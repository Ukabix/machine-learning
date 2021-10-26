Pełna kompilacja w pliku "run.py"  
Dane wejściowe w katalogu "dataset"  
Parametry algorytmu uczącego jak poniżej, 2000 postąpień, 25 epok  
Sieć neuronowa 2 warstwowa, funkcje aktywujące 1x relu + 1x sigmoidalna  

classifier.fit_generator(training_set,
                    steps_per_epoch=2000,
                    epochs=25,
                    validation_data = test_set,
                    validation_steps=2000)
