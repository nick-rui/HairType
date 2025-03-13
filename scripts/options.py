import argparse
import os

class Options:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Input/Output paths
        parser.add_argument("--input_dir", type=str, default='./images/raw_images',
                          help="Directory containing input images")
        parser.add_argument("--output_dir", type=str, default='./images',
                          help="Base directory for all outputs")

        # SAM model settings
        parser.add_argument("--model_type_sam", type=str, default='vit_h',
                          help="SAM model type: ['default', 'vit_h', 'vit_l', 'vit_b']")
        parser.add_argument("--checkpoint_sam", type=str, default='./checkpoints/sam_vit_h_4b8939.pth',
                          help="Path to SAM checkpoint")

        # Image processing settings
        parser.add_argument("--image_size", type=int, default=1024,
                          help="Size for resized images")
        parser.add_argument("--patch_size", type=int, default=256,
                          help="Size of hair patches to extract")
        
        # Contrast enhancement settings
        parser.add_argument("--clahe_clip_limit", type=float, default=3.0,
                          help="Clip limit for CLAHE contrast enhancement")
        parser.add_argument("--clahe_grid_size", type=int, default=8,
                          help="Grid size for CLAHE contrast enhancement")
        
        # Processing flags
        parser.add_argument("--skip_mask_generation", action='store_true',
                          help="Skip mask generation if masks already exist")
        parser.add_argument("--skip_patch_extraction", action='store_true',
                          help="Skip patch extraction if patches already exist")
        parser.add_argument("--skip_contrast_enhancement", action='store_true',
                          help="Skip contrast enhancement if enhanced patches exist")

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = self.initialize(self.parser)

        opt = self.parser.parse_args()
        
        # Create necessary directories
        os.makedirs(opt.output_dir, exist_ok=True)
        opt.resized_dir = os.path.join(opt.output_dir, 'resized_images')
        opt.mask_dir = os.path.join(opt.output_dir, 'hair_masks')
        opt.body_mask_dir = os.path.join(opt.output_dir, 'body_masks')
        opt.masked_dir = os.path.join(opt.output_dir, 'masked_images')
        opt.patches_dir = os.path.join(opt.output_dir, 'hair_patches')
        opt.enhanced_dir = os.path.join(opt.output_dir, 'enhanced_patches')
        
        for directory in [opt.resized_dir, opt.mask_dir, opt.body_mask_dir,
                         opt.masked_dir, opt.patches_dir, opt.enhanced_dir]:
            os.makedirs(directory, exist_ok=True)

        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        return opt