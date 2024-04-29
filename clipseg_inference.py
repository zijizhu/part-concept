def train_epoch_stage1(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                       writer: SummaryWriter, dataset_size: int, epoch: int,
                       device: torch.device, logger: logging.Logger):
    running_ce_loss = 0
    running_proto_loss = 0
    running_total_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        ce_loss, proto_loss, logits = model(images, targets)

        total_loss = ce_loss + 2 * proto_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_ce_loss += ce_loss.item() * len(images)
        running_proto_loss += proto_loss.item() * len(images)
        running_total_loss += total_loss.item() * len(images)
        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    ce_loss_avg = running_ce_loss / dataset_size
    proto_loss_avg = running_proto_loss / dataset_size
    total_loss_avg = running_total_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train/ce', ce_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/proto', proto_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/total', total_loss_avg, epoch)
    writer.add_scalar(f'Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} CE Train Loss: {ce_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Proto Train Loss: {proto_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Total Train Loss: {total_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Aac: {epoch_acc:.4f}')


def train_epoch_stage2(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                       writer: SummaryWriter, dataset_size: int, epoch: int,
                       device: torch.device, logger: logging.Logger):
    running_ce_loss = 0
    running_total_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        ce_loss, logits = model(images, targets)

        total_loss = ce_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_ce_loss += ce_loss.item() * len(images)
        running_total_loss += total_loss.item() * len(images)
        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    ce_loss_avg = running_ce_loss / dataset_size
    total_loss_avg = running_total_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train/ce', ce_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/total', total_loss_avg, epoch)
    writer.add_scalar(f'Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} CE Train Loss: {ce_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Total Train Loss: {total_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Aac: {epoch_acc:.4f}')