#
# # Loop over epochs
# for epoch in range(max_epochs):
#     print(f"epoch {epoch}")
#     # Training
#     precision.reset()
#     recall.reset()
#
#     encoder.train()
#     with torch.set_grad_enabled(True):
#         for batch_idx, (train_X, train_y, train_weights) in enumerate(
#                 itertools.islice(generator_train, dataset_train.n_steps)):
#             subnetwork = train_X["subnetwork"].to(device)
#             input_seqs, train_y, train_weights = train_X["input_seqs"].to(device), train_y.to(device), train_weights.to(
#                 device)
#             optimizer.zero_grad()
#
#             # Model computations
#             Y_hat = encoder(input_seqs, subnetwork)
#             loss = encoder.loss(Y_hat, train_y, None)
#             loss.backward()
#             optimizer.step()
#
#             update(Y_hat, train_y)
#     print(f"\t loss {loss.item():.3f}, precision: {precision.compute():.3f}, recall: {recall.compute():.3f}")
#
#     # Validation
#     precision.reset()
#     recall.reset()
#
#     encoder.eval()
#     with torch.set_grad_enabled(False):
#         for batch_idx, (test_X, test_y, test_weights) in enumerate(
#                 itertools.islice(generator_test, dataset_test.n_steps)):
#             subnetwork = test_X["subnetwork"].to(device)
#             input_seqs, test_y, test_weights = test_X["input_seqs"].to(device), test_y.to(device), test_weights.to(
#                 device)
#
#             Y_hat = encoder(input_seqs, subnetwork)
#             loss = encoder.loss(Y_hat, test_y, None)
#
#             update(Y_hat, test_y)
#     print(f"\t val_loss {loss.item():.3f}, precision: {precision.compute():.3f}, recall: {recall.compute():.3f}")
#
