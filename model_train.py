def train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, path, early_stop):
    """
    pytorch 模型训练通用代码
    :param model: pytorch 模型
    :param train_loader: dataloader, 训练数据
    :param val_loader: dataloader, 验证数据
    :param epoch: int, 训练迭代次数
    :param loss_function: 优化损失函数
    :param optimizer: pytorch优化器
    :param path: save path
    :param early_stop: int, 提前停止步数
    :return: None
    """
    # 是否使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
    model = model.to(device)
    
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    
    # 训练
    for i in range(epoch):
        total_loss, count = 0, 0
        y_pred = list()
        y_true = list()
        for idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y = x.to(device), y.to(device) 
            u, m = model(x)
            predict = torch.sigmoid(torch.sum(u*m, 1))
            y_pred.extend(predict.cpu().detach().numpy())
            y_true.extend(y.cpu().detach().numpy())
            loss = loss_function(predict, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
            
        train_auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        torch.save(model, path.format(i+1))
        print("Epoch %d train loss is %.3f and train auc is %.3f" % (i+1, total_loss / count, train_auc))
    
        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        val_y_pred = list()
        val_true = list()
        for idx, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y = x.to(device), y.to(device)
            u, m = model(x)
            predict = torch.sigmoid(torch.sum(u*m, 1))
            val_y_pred.extend(predict.cpu().detach().numpy())
            val_true.extend(y.cpu().detach().numpy())
            loss = loss_function(predict, y.float())
            total_eval_loss += float(loss)
            count_eval += 1
        val_auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        print("Epoch %d val loss is %.3fand train auc is %.3f" % (i+1, total_eval_loss / count_eval, val_auc))
        
        # 提前停止策略
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break
