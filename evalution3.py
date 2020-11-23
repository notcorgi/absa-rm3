import matplotlib.pyplot as plt

'''
mse-rmsp;batch-size=256;路透社数据集
'''
list_loss_orig = [0.018932770085514705, 0.012863765509167027, 0.009933313324546977, 0.00858473284687369,
                  0.007851052113530895, 0.007329553995384234, 0.006887520251586641, 0.006516929082968817,
                  0.006173569072299715, 0.0058277242519256395, 0.0055014205568592305, 0.005199616842377909,
                  0.004903483624520181, 0.00463762482669475, 0.004413937493171541]

list_loss_50_10epoch = []
list_loss_100_10epoch = [0.018483672273296555, 0.011047660940033285, 0.008438491733399311, 0.007246656427336192,
                         0.006483017819074398, 0.0058909345036376, 0.005441430373573368, 0.00500604097284243,
                         0.004689742092570408, 0.004353644706670171, 0.004067889966458252, 0.0038136394739268064,
                         0.0036146686315378877, 0.0034454835187454807, 0.00334934539886471]

list_loss_200_10epoch = [0.01757644540746639, 0.009092852242701893, 0.00787094296653851, 0.006915988841006625, 0.006162280122275147, 0.005470847510657241, 0.004948552454437276, 0.004516997614366855, 0.004176861448975062, 0.003907578931177745, 0.0036607803644357335, 0.0034633693165745764, 0.003341847334758091, 0.0032193107550004436, 0.0031415132530790355]


list_loss_orig_val = [0.01548558847980783, 0.011966332244087379, 0.009935409873848298, 0.009294165258976455,
                      0.009052181510758215, 0.008864288152018365, 0.008635361427161531, 0.00852046789470915,
                      0.008343914003153896, 0.008335838835578707, 0.008234412799401828, 0.008114624159872632,
                      0.007991789366212882, 0.00796438119342101, 0.007869404693495678]

list_loss_50_10epoch_val = []
list_loss_100_10epoch_val = [0.015378059972313806, 0.010589519110459574, 0.009095619439879831, 0.008647971688143005,
                             0.00834979857416519, 0.008147184018592648, 0.008024901742668816, 0.008061582967949756,
                             0.007829409872220043, 0.007780491011331192, 0.007513784153906999, 0.007608498876046253,
                             0.00740462759907489, 0.007348019762836952, 0.007367414421815263]

list_loss_200_10epoch_val = [0.014888444678404174, 0.009752820272763592, 0.008805745835514434, 0.00840433274662004, 0.008201856707410268, 0.007879026416477823, 0.00792035569506257, 0.007731913154358428, 0.007668087806436622, 0.007535302763425634, 0.007673611048115904, 0.007506374887670406, 0.007522053175913512, 0.007420412093313935, 0.007450232415918977]


list_acc_orig = [0.4800198, 0.61239636, 0.6881108, 0.71805024, 0.7387109, 0.75578374, 0.7670419, 0.77397007, 0.7846097,
                 0.79636276, 0.81417793, 0.8279104, 0.8402821, 0.84968454, 0.85426205]

list_acc_50_10epoch = []
list_acc_100_10epoch = [0.49894843, 0.6629964, 0.7378449, 0.7769393, 0.7948781, 0.8129407, 0.8206112, 0.8373129,
                        0.8448596, 0.85822093, 0.8714586, 0.8795002, 0.88308793, 0.8849437, 0.8863046]

list_acc_200_10epoch = [0.52876407, 0.70407027, 0.74205124, 0.7758258, 0.80081654, 0.821106, 0.8420141, 0.85438573, 0.86626256, 0.87529385, 0.87999505, 0.8849437, 0.8901398, 0.8910058, 0.89174813]


list_acc_orig_val = [0.538375973701477, 0.649610698223114, 0.6896551847457886, 0.7063403725624084, 0.7063403725624084,
                     0.7085650563240051, 0.7163515090942383, 0.7163515090942383, 0.7285873293876648, 0.7263626456260681,
                     0.7274749875068665, 0.7452725172042847, 0.7541713118553162, 0.7519466280937195, 0.7575083374977112]

list_acc_50_10epoch_val = []
list_acc_100_10epoch_val = [0.5984427332878113, 0.6818687319755554, 0.7230255603790283, 0.7285873293876648,
                            0.7296996712684631, 0.7363737225532532, 0.7408231496810913, 0.7441601753234863,
                            0.7530589699745178, 0.7586206793785095, 0.774193525314331, 0.770856499671936,
                            0.7853170037269592, 0.7808676362037659, 0.7808676362037659]

list_acc_200_10epoch_val = [0.6051835441589355, 0.7056663212776184, 0.7219021415710449, 0.7285873293876648, 0.7352613806724548, 0.7597330212593079, 0.7519466280937195, 0.7664071321487427, 0.7652947902679443, 0.7786429524421692, 0.7664071321487427, 0.7764182686805725, 0.7697441577911377, 0.7753058671951294, 0.770856499671936]


list_epoch = [x for x in range(1, 16)]
plt.figure()
# 建立 subplot 网格，高为 2，宽为 1
# 激活第一个 subplot
plt.subplot(2, 2, 1)
plt.plot(list_epoch, list_loss_orig, linestyle=':', color='b', label='orig')  # ls代表线类型，：为虚线，默认是实线
# plt.plot(list_epoch, list_loss_50_10epoch, linestyle='--', color='black', label='alpha=10')  # ls代表线类型，：为虚线，默认是实线
plt.plot(list_epoch, list_loss_100_10epoch, linestyle='--', color='y', label='alpha=50')  # ls代表线类型，：为虚线，默认是实线
plt.plot(list_epoch, list_loss_200_10epoch, linestyle='--', color='r', label='alpha=500')  # ls代表线类型，：为虚线，默认是实线
plt.title("mse-rmsp;batch-size=256;epoch=15;train-loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.legend()
print('-----------------------------------------')
plt.subplot(2, 2, 2)
plt.plot(list_epoch, list_acc_orig, linestyle=':', color='b', label='orig')  # ls代表线类型，：为虚线，默认是实线
# plt.plot(list_epoch, list_acc_50_10epoch, linestyle='--', color='black', label='alpha=10')  # ls代表线类型，：为虚线，默认是实线
plt.plot(list_epoch, list_acc_100_10epoch, linestyle='--', color='y', label='alpha=50')  # ls代表线类型，：为虚线，默认是实线
plt.plot(list_epoch, list_acc_200_10epoch, linestyle='--', color='r', label='alpha=500')  # ls代表线类型，：为虚线，默认是实线
plt.title("mse-rmsp;batch-size=256;epoch=15;train-acc")
plt.xlabel("epoch")
plt.ylabel("train_acc")
plt.legend()
print('------------------------------------------')
plt.subplot(2, 2, 4)
plt.plot(list_epoch, list_acc_orig_val, linestyle=':', color='b', label='orig')  # ls代表线类型，：为虚线，默认是实线
# plt.plot(list_epoch, list_acc_50_10epoch_val, linestyle='--', color='black', label='alpha=10')  # ls代表线类型，：为虚线，默认是实线
plt.plot(list_epoch, list_acc_100_10epoch_val, linestyle='--', color='y', label='alpha=50')  # ls代表线类型，：为虚线，默认是实线 -
plt.plot(list_epoch, list_acc_200_10epoch_val, linestyle='--', color='r', label='alpha=500')  # ls代表线类型，：为虚线，默认是实线 -
plt.title("mse-rmsp;batch-size=256;epoch=15;val-acc")
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.legend()

print('------------------------------------------')
plt.subplot(2, 2, 3)
plt.plot(list_epoch, list_loss_orig_val, linestyle=':', color='b', label='orig')  # ls代表线类型，：为虚线，默认是实线
# plt.plot(list_epoch, list_loss_50_10epoch_val, linestyle='--', color='black', label='alpha=10')  # ls代表线类型，：为虚线，默认是实线
plt.plot(list_epoch, list_loss_100_10epoch_val, linestyle='--', color='y', label='alpha=50')  # ls代表线类型，：为虚线，默认是实线 -
plt.plot(list_epoch, list_loss_200_10epoch_val, linestyle='--', color='r', label='alpha=500')  # ls代表线类型，：为虚线，默认是实线 -
plt.title("mse-rmsp;batch-size=256;epoch=15;val-loss")
plt.xlabel("epoch")
plt.ylabel("val_loss")
plt.legend()

plt.show()
