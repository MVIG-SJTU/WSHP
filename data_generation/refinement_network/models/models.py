def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pix':
        assert(opt.dataset_mode.find('aligned') != -1)
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode.find('single') != -1)
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
