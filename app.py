import os
import json
from flask import Flask, request, url_for, redirect, render_template
from function import *

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

im1_path = ''
im2_path = ''
flag=1
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static/upload_image'),
)

@app.route('/', methods=['POST', 'GET'])
def upload():
    global im1_path, im2_path, flag
    
    if request.method == 'POST' :
        f = request.files.get('file')
        file_path = os.path.join(app.config['UPLOADED_PATH'], f.filename)
        f.save(file_path)
        if flag==1:
            im1_path = file_path
            flag=2
        else:
            im2_path = file_path
            flag=1
        
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    # show the form, it wasn't submitted
    patch_size = 7

    im1 = io.imread(im1_path)[:,:,0].astype(np.float32)
    im2 = io.imread(im2_path)[:,:,0].astype(np.float32)
    im_di = dicomp(im1, im2)
    ylen, xlen = im_di.shape
    pix_vec = im_di.reshape([ylen*xlen, 1])

    preclassify_lab = hcluster(pix_vec, im_di)
    print('... ... hiearchical clustering finished !!!')

    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:,:,0] = im1
    mdata[:,:,1] = im2
    mdata[:,:,2] = im_di
    mlabel = preclassify_lab

    x_test = createTestingCubes(mdata, patch_size)
    x_test = x_test.transpose(0, 3, 1, 2)

    ############################################################################################################################################
    # 逐像素预测类别
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    istrain=False
    net=torch.load('./model/self_model.pth',map_location="cuda:0")
    net.to(device)
    net.eval()
    outputs = np.zeros((ylen, xlen))
    glo_fin=torch.Tensor([]).cuda()
    dct_fin=torch.Tensor([]).cuda()
    for i in range(ylen):
        for j in range(xlen):
            if preclassify_lab[i, j] != 1.5 :
                outputs[i, j] = preclassify_lab[i, j]
            else:
                img_patch = x_test[i*xlen+j, :, :, :]
                img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
                img_patch = torch.FloatTensor(img_patch).to(device)
                prediction = net(img_patch)

                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i, j] = prediction+1
        if (i+1) % 50 == 0:
            print('... ... row', i+1, ' handling ... ...')

    outputs = outputs-1
    res = outputs*255
    res = postprocess(res)
    plt.imshow(res,'gray')
    plt.savefig('./static/output/output.png')
    return render_template('result.html')

@app.route('/cool_form', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
    # do stuff when the form is submitted

    # redirect to end the POST handling
    # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

  # show the form, it wasn't submitted
    return render_template('submit.html')

if __name__ == '__main__':
    app.run(debug=True)