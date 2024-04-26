from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, ITKReader
import torch
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
)
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from utils.visulization import display_current_results
from monai.metrics import DiceMetric
import SimpleITK as sitk
import torch.nn.functional as F
from utils.inferer import mirror_sliding_window_inference
from utils.log import Logger


def save_array(array, filename):
    array = array.detach().cpu().numpy()
    sitk.WriteImage(sitk.GetImageFromArray(array), filename)


def cardio_vessel_segmentation_train(cfg, model, num_class, loss_function, val_metric, optimizer,
                                     lr_scheduler, train_dataset, val_dataset, experiment_path,
                                     device, metric_record, start_epoch=0, sw_batch_size=4, mirror_axes=None,
                                     overlap=0.25):
    """
    :param cfg:
    :param model:
    :param loss_function:
    :param val_metric:
    :param optimizer:
    :param train_dataset:
    :param val_dataset:
    :param experiment_path:
    :param device:
    :return:
    """
    printer = Logger(log_dir=experiment_path)
    writer = SummaryWriter(log_dir=os.path.join(experiment_path, 'tensorboard_record')) \
        if cfg.dic['visual']['use_tensorboard'] else False
    index = cfg.dic['visual']['show_image_index']
    model_save_path = os.path.join(experiment_path, 'checkpoint')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_dataset, batch_size=cfg.dic['batch_size'], shuffle=True, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)

    max_epochs = cfg.dic['iters']
    iters_each_epoch = 100
    val_interval = cfg.dic['train']['val_interval']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    post_pred_one_hot = Compose([AsDiscrete(argmax=True, to_onehot=num_class)])
    post_pred = Compose([AsDiscrete(argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=num_class)])

    model.to(device)
    for epoch in range(start_epoch, max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        while step < iters_each_epoch:
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if step % int(iters_each_epoch/10) == 0:
                    printer.p(
                        f"{step}/{iters_each_epoch}, "
                        f"train_loss: {loss.item():.4f},"
                        f"lr:{optimizer.param_groups[0]['lr']}")
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        printer.p(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if writer:
            writer.add_scalars("train_loss", {"train_loss": epoch_loss}, epoch)
            writer.add_scalars("learning_rate", {"lr": optimizer.param_groups[0]['lr']}, epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = tuple(cfg.dic['transform']['patch_size'])
                    val_outputs = mirror_sliding_window_inference(inputs=val_inputs, roi_size=roi_size,
                                                                  sw_batch_size=sw_batch_size, predictor=model,
                                                                  mirror_axes=mirror_axes, overlap=overlap)
                    val_outputs = [post_pred_one_hot(one_output) for one_output in decollate_batch(val_outputs)]
                    val_labels = [post_label(one_label) for one_label in decollate_batch(val_labels)]
                    # val_outputs = [post_pred(one_output) for one_output in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    # try:
                    #     val_metric(y_pred=val_outputs, y=val_labels)
                    # except ValueError:
                    #     print(val_inputs[0][0].meta['filename_or_obj'])
                    # else:
                    #     print('unexpected Error is found')
                    #     raise RuntimeError
                    # print(f"image:{val_data['image'][0].shape}, label{val_data['label'][0].shape}")
                    # print(val_inputs[0][0].meta['filename_or_obj'], val_outputs[0].shape, val_labels[0].shape, )
                    val_metric(y_pred=val_outputs, y=val_labels)
                    # if writer and i in index:
                    #     display_current_results(
                    #         writer=writer,
                    #         image=val_data["image"][0, 0, :, :, val_data["image"].shape[-1]//2],
                    #         label=val_data["label"][0, 0, :, :, val_data["image"].shape[-1]//2],
                    #         output=torch.argmax(val_outputs[0], dim=0).detach().cpu()[:, :, val_data["image"].shape[-1]//2],
                    #         image_index=i,
                    #         image_name=os.path.split(val_data["image"].meta['filename_or_obj'][0])[-1][:-7]+f"_epoch{epoch}",
                    #         save_visuals=False)
                # aggregate the final mean dice result
                metric = val_metric.aggregate().item()
                val_metric.reset()
                if writer:
                    writer.add_scalars("val_metric", {"val_loss": metric}, epoch)
                    writer.add_text("loss", f"epoch {epoch} average loss: "
                                            f"{epoch_loss:.4f}, val metric: {metric}", epoch)



                # reset the status for next validation round

                metric_values.append(metric)
                if metric > best_metric:
                    # save gif in tensorboard
                    # if writer:
                    #     plot_2d_or_3d_image(val_inputs, step=epoch, writer=writer, index=0, max_channels=1,
                    #                         frame_dim=-3, max_frames=24, tag='input')
                    #     plot_2d_or_3d_image(val_outputs, step=epoch, writer=writer, index=0, max_channels=1,
                    #                         frame_dim=-3, max_frames=24, tag='predict_label')
                    #     plot_2d_or_3d_image(val_labels, step=epoch, writer=writer, index=0, max_channels=1,
                    #                         frame_dim=-3, max_frames=24, tag='ground_truth')
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_save_path, "best_metric_model.pth"))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state': optimizer.state_dict(),
                        'lr': optimizer.param_groups[0]['lr']
                    }, os.path.join(model_save_path, "best_metric_model_training_stage.pth"))
                    # print("saved new best metric model")
                printer.p(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" best mean dice: {best_metric:.4f} "
                    f" at epoch: {best_metric_epoch}"
                )
    if writer:
        writer.close()

    # plot the loss and dice curve
    # plt.figure("train", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Epoch Average Loss")
    # x = [i + 1 for i in range(len(epoch_loss_values))]
    # y = epoch_loss_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.subplot(1, 2, 2)
    # plt.title("Val Mean Dice")
    # x = [val_interval * (i + 1) for i in range(len(metric_values))]
    # y = metric_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.show()
    # plt.savefig(os.path.join(experiment_path, f"loss_record_{fold_order}.png"))
    # plt.close()

    # save the metric
    metric_record.append(best_metric)


def cardio_vessel_segmentation_multi_phase_with_image_train(cfg, model, key, num_class, loss_function, val_metric, optimizer,
                                                            lr_scheduler, train_dataset, val_dataset, experiment_path,
                                                            device, start_epoch=0, sw_batch_size=4, mirror_axes=None,
                                                            overlap=0.25, debug=False,):
    """
    :param cfg:
    :param model:
    :param: which information to use
    :param loss_function:
    :param val_metric:
    :param optimizer:
    :param train_dataset:
    :param val_dataset:
    :param experiment_path:
    :param device:
    :param debug: whether save the image for checking during training
    :return:
    """
    printer = Logger(log_dir=experiment_path)
    origin_metric = DiceMetric(include_background=False, reduction="mean")
    rs_metric = DiceMetric(include_background=False, reduction="mean")
    val_metric_not_merge = DiceMetric(include_background=False, reduction="mean")
    writer = SummaryWriter(log_dir=os.path.join(experiment_path, 'tensorboard_record')) \
        if cfg.dic['visual']['use_tensorboard'] else False
    index = cfg.dic['visual']['show_image_index']
    model_save_path = os.path.join(experiment_path, 'checkpoint')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_dataset, batch_size=cfg.dic['batch_size'], shuffle=True, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)

    max_epochs = cfg.dic['iters']
    val_interval = cfg.dic['train']['val_interval']
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred_one_hot = Compose([AsDiscrete(argmax=True, to_onehot=num_class)])
    post_pred = Compose([AsDiscrete(argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=num_class)])

    model.to(device)
    print(key)
    for epoch in range(start_epoch, max_epochs):
        printer.p("-" * 10)
        printer.p(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = torch.zeros((batch_data[key[0]].shape[0], len(key), batch_data[key[0]].shape[2],
                                  batch_data[key[0]].shape[3], batch_data[key[0]].shape[4]))
            for j in range(len(key)):
                inputs[:, j:j + 1, :, :, :] = batch_data[key[j]]
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs * batch_data['CS_DLGT'].to(device),
                                 batch_data['label'].to(device) * batch_data['CS_DLGT'].to(
                                     device))  # + loss_function(outputs * attention, labels * attention) * 0.5
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            printer.p(
                f"{step}/{len(train_loader)}, "
                f"train_loss: {loss.item():.4f},"
                f"lr:{optimizer.param_groups[0]['lr']}")

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        printer.p(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if writer:
            writer.add_scalars("train_loss", {"train_loss": epoch_loss}, epoch)
            writer.add_scalars("learning_rate", {"lr": optimizer.param_groups[0]['lr']}, epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, val_data in enumerate(val_loader):
                    cs_m = val_data['CS_M'].to(device)
                    cs_dl = val_data['CS_DL'].to(device)
                    cs_dlgt = val_data['CS_DLGT'].to(device)
                    i_m = val_data['I_M'].to(device)
                    val_labels = val_data['label'].to(device)
                    val_inputs = torch.zeros((val_data[key[0]].shape[0], len(key), val_data[key[0]].shape[2],
                                              val_data[key[0]].shape[3], val_data[key[0]].shape[4]), device=device)
                    for j in range(len(key)):
                        val_inputs[:, j, :, :, :] = val_data[key[j]].to(device)

                    roi_size = tuple(cfg.dic['transform']['patch_size'])
                    val_outputs = mirror_sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,
                                                                  mirror_axes=mirror_axes, overlap=overlap,
                                                                  mode='gaussian')
                    val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                    rcs_gt = val_outputs * cs_dlgt + cs_m * (1 - cs_dlgt)
                    rcs_gt = [one_output for one_output in decollate_batch(rcs_gt)]
                    rcs = val_outputs * cs_dl + cs_m * (1 - cs_dl)
                    rcs = [one_output for one_output in decollate_batch(rcs)]
                    if debug:
                        cs_a = val_data['CS_A'].to(device)
                        i_a = val_data['I_A'].to(device)
                        val_outputs = [one_output for one_output in decollate_batch(val_outputs)]
                        val_labels = [one_label for one_label in decollate_batch(val_labels)]
                        cs_m = [one_input for one_input in decollate_batch(cs_m)]
                        cs_a = [one_input for one_input in decollate_batch(cs_a)]
                        cs_dlgt = [one_input for one_input in decollate_batch(cs_dlgt)]
                        cs_dl = [one_input for one_input in decollate_batch(cs_dl)]
                        path = os.path.join(experiment_path, 'debug_img')
                        if not os.path.exists(path):
                            os.mkdir(path)
                        name = os.path.split(cs_m[0][0].meta['filename_or_obj'])[-1]
                        save_array(val_outputs[0][0], os.path.join(path, f'{name}_RS.nii.gz'))
                        save_array(rcs[0][0], os.path.join(path, f'{name}_RCS.nii.gz'))
                        save_array(val_labels[0][0], os.path.join(path, f'{name}_GT.nii.gz'))
                        save_array(cs_m[0][0], os.path.join(path, f'{name}_CS_M.nii.gz'))
                        save_array(cs_a[0][0], os.path.join(path, f'{name}_CS_A.nii.gz'))
                        save_array(cs_dl[0][0], os.path.join(path, f'{name}_CS_DLGT.nii.gz'))
                        save_array(cs_dlgt[0][0], os.path.join(path, f'{name}_CS_DL.nii.gz'))
                        save_array(rcs_gt[0][0], os.path.join(path, f'{name}_RCSGT.nii.gz'))

                    val_metric(y_pred=rcs_gt, y=val_labels)
                    origin_metric(y_pred=cs_m, y=val_labels)

                # aggregate the final mean dice result
                metric = val_metric.aggregate().item()
                origin_metric_value = origin_metric.aggregate().item()
                if writer:
                    writer.add_scalars("val_metric", {"val_loss": metric}, epoch)
                    writer.add_text("loss", f"epoch {epoch} average loss: "
                                            f"{epoch_loss:.4f}, val metric: {metric}", epoch, )

                # reset the status for next validation round
                val_metric.reset()
                origin_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:

                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_save_path, "best_metric_model.pth"))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state': optimizer.state_dict(),
                        'lr': optimizer.param_groups[0]['lr']
                    }, os.path.join(model_save_path, "best_metric_model_training_stage.pth"))
                    # print("saved new best metric model")
                printer.p(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" best mean dice: {best_metric:.4f} "
                    f" origin_metric: {origin_metric_value:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
    if writer:
        writer.close()