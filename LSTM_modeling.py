from LSTM_preprocessing import dengue_dataloader


#讀取處理好的loader可修改參數window_size、batch_size、split_year
train_loader,test_loader,weight,dim=dengue_dataloader(window_size=4,
                                                      batch_size=64,
                                                      split_year=2024)
