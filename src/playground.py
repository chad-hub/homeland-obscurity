# %%
'''attempt to visualize CNN layers with callbacks '''


  # test_images = []
  # test_labels = []
  # for idx,i in enumerate(range(0, 4, 3)):
  #   for j in range(6):
  #     test_images.append(validation_generator.__getitem__(i)[0][j])
  #     test_labels.append(validation_generator.__getitem__(i)[1][j])

  # validation_class_zero = (
  #     np.array(
  #         [
  #             el
  #             for el, label in zip(test_images, test_labels)
  #             if np.all(np.argmax(label) == 0)
  #         ][0:5]
  #     ),
  #     None,
  # )
  # validation_class_five = (
  #     np.array(
  #         [
  #             el
  #             for el, label in zip(test_images, test_labels)
  #             if np.all(np.argmax(label) == 5)
  #         ][0:5]
  #     ),
  #     None,
  # )
  # callbacks = [

  #     tf_explain.callbacks.GradCAMCallback(
  #         validation_class_zero, class_index=0
  #     )
      # tf_explain.callbacks.GradCAMCallback(
      #     validation_class_five, class_index=5
      # ),
      # # tf_explain.callbacks.ActivationsVisualizationCallback(
      # #     validation_class_zero
      
      # tf_explain.callbacks.SmoothGradCallback(
      #     validation_class_zero, class_index=0, num_samples=15, noise=1.0
      # ),
      # tf_explain.callbacks.IntegratedGradientsCallback(
      #     validation_class_zero, class_index=0, n_steps=10
      # ),
      # tf_explain.callbacks.VanillaGradientsCallback(validation_class_zero, class_index=0),
      # tf_explain.callbacks.GradientsInputsCallback(validation_class_zero, class_index=0),
  # ]