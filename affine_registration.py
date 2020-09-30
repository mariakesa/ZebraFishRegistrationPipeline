
import javabridge as jv
import bioformats as bf
from xml import etree as et
import torch as th
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from threading import Thread, Lock
from queue import Queue
from pyprind import prog_percent


class Pyfish:

    def __init__(self, lif_file_path='C:/Users/koester_lab/Documents/Maria/data/fish20_6dpf_medium.lif', save_path="", align_to_frame=0, use_gpu=True, max_displacement=300, thread_count=4):
        lif_file_path='C:/Users/koester_lab/Documents/Maria/data/fish20_6dpf_medium.lif'
        self.lif_file_path = lif_file_path
        self.align_to_frame = align_to_frame #un frami ke nesbat be un baghie axaro align mikonim
        self.use_gpu = use_gpu
        self.max_displacement = max_displacement
        self.thread_count = thread_count

        self._start_lif_reader()
        self._set_shapes()
        self._prepare_alignment_frame()

    @staticmethod
    def _to_gpu(numpy_array):
        return Variable(th.from_numpy(numpy_array.astype(np.float32)).cuda(), requires_grad=False)

    @staticmethod
    def _to_cpu(pytorch_tensor):
        return pytorch_tensor.cpu().numpy()

    def _start_lif_reader(self):
        jv.start_vm(class_path=bf.JARS)

        log_level = 'ERROR'
	    # reduce log level

        # currently does not work in new conda environment

        #rootLoggerName = jv.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
        #rootLogger = jv.static_call("org/slf4j/LoggerFactory", "getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
        #logLevel = jv.get_static_field("ch/qos/logback/classic/Level", log_level, "Lch/qos/logback/classic/Level;")
        #jv.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

        self.ir = bf.ImageReader(self.lif_file_path, perform_init=True)
        mdroot = et.ElementTree.fromstring(bf.get_omexml_metadata(self.lif_file_path))
        mds = list(map(lambda e: e.attrib, mdroot.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')))

        # lif can contain multiple images, select one that is likely to be the timeseries
        self.metadata = None
        self.lif_stack_idx = 0
        for idx, md in enumerate(mds):
            if int(md['SizeT']) > 1:
                self.lif_stack_idx = idx
                self.metadata      = md
        print('lif stack: ', self.lif_stack_idx)
        if not self.metadata: raise ValueError('lif does not contain an image with sizeT > 1')


    def _set_shapes(self):
        self.nt = int(self.metadata['SizeT'])
        self.nz = int(self.metadata['SizeZ'])
        self.nx = int(self.metadata['SizeX'])
        self.ny = int(self.metadata['SizeY'])
        #print ('self.nt', self.nt)

        #self.nt = 300


        self.stack_shape = (self.nt, self.nz, self.nx, self.ny)
        self.frame_shape = (self.nz, self.nx, self.ny)
        self.plane_shape = (self.nx, self.ny)
        self.frame_dims = 3
        self.plane_dims = 2

    def _prepare_alignment_frame(self):

        # read alignment frame
        self.alignment_frame = self.read_frame(self.align_to_frame)

        # upload to GPU and prepare for cross correlation
        if self.use_gpu:
            self.frame_midpoints = np.array([np.fix(axis_size / 2) for axis_size in self.frame_shape])
            self.affine_grid_size = th.Size([1, 1, self.frame_shape[0], self.frame_shape[1], self.frame_shape[2]])
            #print (self.alignment_frame)
            frame_tensor = self._to_gpu(self.alignment_frame) #frame_tensor majmue plane ha dar z haye mokhtalef baraye t=0 (alignment frame) hast.
            self.alignment_frame_fourier = th.rfft(frame_tensor, self.frame_dims, onesided=False)
                                                     #for plane in frame_tensor])
            self.alignment_frame_fourier[:,:,:,1] *= -1        # complex conjugate for crosscorrelation

    def read_frame(self, t):
        frame = np.empty(self.frame_shape, dtype=np.uint16)
        for z in range(self.nz):
            frame[z] = self._read_plane(t, z)
        #frame = np.swapaxes(frame,1,2)
        return frame

    def _read_plane(self, t, z):
        return self.ir.read(t=t, z=z, c=0, series=self.lif_stack_idx, rescale=False)

    def register_whole_stack(self, save_path="", max_cell_radius=5, cell_size_iterations=5, cell_rel_threshold=0.4, min_cell_intensity=20,
                            cell_overlap=0):

        #disp = []
        #self.aligned_frame = [None] * self.nt
        self.displacement = np.empty([self.nt, 3])
        self.registered_stack = np.empty((self.nt, self.nz, self.nx, self.ny),dtype=np.uint16)
        self._prepare_std_deviation_and_invalid_frames_and_result()
        self._prepare_invalid_frames()

        register_queue = self._create_queue(self._align_frame_worker, self.thread_count * 2)

        for t in prog_percent(range(self.nt)):
            register_queue.put([self.read_frame(t), t, self.registered_stack[t], t])
        register_queue.join()

        return self.displacement, self.registered_stack

    def _prepare_std_deviation_and_invalid_frames_and_result(self):
        if self.use_gpu:
            self.std_deviation_sum    = self._to_gpu(np.empty(self.frame_shape))
            self.std_deviation_sum_sq = self._to_gpu(np.empty(self.frame_shape))
        else:
            self.std_deviation_sum    = np.empty(self.frame_shape)
            self.std_deviation_sum_sq = np.empty(self.frame_shape)
        self.std_deviation_sum_mutex = Lock()

    def _prepare_invalid_frames(self):
        self.invalid_frames = np.empty(0)

    def _align_frame_worker(self, queue):
        while(True):
            #frame, t, dst, dst_idx = queue.get()
            frame, t, dst, dst_idx = queue.get()
            #print ('dst.idx',dst_idx)
            #dst[dst_idx] = self._align_frame(frame, t)
            dst = self._align_frame(frame, t)
            self.registered_stack[t] = dst
            queue.task_done()
        return

    def _align_frame(self, frame, t):

        shifts = np.zeros([1, 3])
        if self.use_gpu:
            frame_tensor = self._to_gpu(frame)

            ## 3D translation and warping here

            shift = self._register_translation_gpu(frame_tensor)
            if np.sqrt(np.sum(shift**2)) > self.max_displacement:
                 self.invalid_frames = np.append(self.invalid_frames, t)
                 frame_tensor[:,:,:] = 0
                 #self.registered_stack[t] = frame_tensor
                 self.displacement[t] = shifts
                 return self._to_cpu(frame_tensor)

            shifts = shift
            self.displacement[t] = shifts
            #if np.absolute (shifts[0]) > 1:
             #  shifts[0] = np.where(shifts[0] > 0, [0.99],[-.99])
            aligned_frame_tensor = self._warp_gpu(frame_tensor, shifts)
            with self.std_deviation_sum_mutex:
                self.std_deviation_sum    += aligned_frame_tensor
                self.std_deviation_sum_sq += aligned_frame_tensor**2
            aligned_frame_tensor = self._to_cpu(aligned_frame_tensor)
            #self.registered_stack[t] = aligned_frame_tensor
            return aligned_frame_tensor
        '''
        else:
            #for z in range(self.nz):
                #shift, _, _ = register_translation(frame[z], alignment_frame[z])
            shift = self._register_translation_cpu(frame, self.alignment_frame, upsample_factor=20)
            if np.sqrt(np.sum(shift**2)) > self.max_displacement:
                self.invalid_frames = np.append(self.invalid_frames, t)
                frame_tensor[:,:,:] = 0
                #self.registered_stack[t] = frame_tensor
                return frame_tensor.numpy()
                #shifts = shift
            #self.displacement[t] = shift
            print (shift)
            shift = self._to_gpu(shift)
            #frame_tensor = self._to_gpu(frame)
            #return shift
            frame_tensor = self._to_gpu(frame)
            aligned_frame_tensor = self._warp_gpu(frame_tensor, shift)
            self.registered_stack[t] = frame_tensor
            #aligned_frame_tensor = self._warp_gpu(frame, shift)
           # for z in range(self.nz):
              #  aligned_frame[z] = warp(aligned_frame[z], AffineTransform(translation=shifts[z]), preserve_range=True)
            with self.std_deviation_sum_mutex:
                self.std_deviation_sum    += aligned_frame_tensor
                self.std_deviation_sum_sq += aligned_frame_tensor**2
            return self._to_cpu(aligned_frame_tensor)
        '''

    def _register_translation_gpu(self,frame,upsample_factor=20):
        # Whole-pixel shift - Compute cross-correlation by an IFFT
        img_fourier = th.rfft(frame, self.frame_dims, onesided=False)
        image_product = th.zeros(img_fourier.size())
        image_product[:,:,:,0] = img_fourier[:,:,:,0]* self.alignment_frame_fourier[:,:,:,0]- \
        img_fourier[:,:,:,1]* self.alignment_frame_fourier[:,:,:,1]
        image_product[:,:,:,1] = img_fourier[:,:,:,0]* self.alignment_frame_fourier[:,:,:,1]+ \
        img_fourier[:,:,:,1]* self.alignment_frame_fourier[:,:,:,0]
        cross_correlation = th.irfft(image_product, self.frame_dims, onesided=False, signal_sizes = frame.shape)
        # Locate maximum
        maxima = self._to_cpu(th.argmax(cross_correlation))
        maxima = np.unravel_index(maxima, cross_correlation.size(), order='C')
        maxima = np.asarray(maxima)
        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > self.frame_midpoints] -= np.array(cross_correlation.shape)[shifts > self.frame_midpoints] # in bara chie??
        shifts = np.round(shifts * upsample_factor) / upsample_factor #aya round numpy ba torch yejur amal mikone?bale
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (img_fourier.numel() * upsample_factor ** 2)
        sample_region_offset = dftshift - shifts*upsample_factor
        image_product = self._to_cpu(image_product)
        imag_part = 1j*image_product[:,:,:,1]
        img_product_cpu = image_product[:,:,:,0]+imag_part
        cross_correlation = self._upsampled_dft_cpu(img_product_cpu.conj(), upsampled_region_size, upsample_factor, sample_region_offset).conj()

        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)),cross_correlation.shape,order='C'), dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        return shifts

    # adapted from: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation
    def _upsampled_dft_cpu(self,data, upsampled_region_size, upsample_factor=20, axis_offsets=None):
        #print ('data', data.shape)
        if not hasattr(upsampled_region_size, "__iter__"):
            upsampled_region_size = [upsampled_region_size, ] * data.ndim
        else:
            if len(upsampled_region_size) != data.ndim:
                raise ValueError("shape of upsampled region sizes must be equal to input data's number of dimensions.")
        if axis_offsets is None:
            axis_offsets = [0, ] * data.ndim
        else:
            if len(axis_offsets) != data.ndim:
                raise ValueError("number of axis offsets must be equal to input data's number of dimensions.")
        if data.ndim == 3:
            im2pi = 1j * 2 * np.pi
            dim_kernels = []
            #print ('upsampled_region_size',upsampled_region_size)
            #print('axis_offsets',axis_offsets)
            for (n_items, ups_size, ax_offset) in zip(data.shape,upsampled_region_size, axis_offsets):
                dim_kernels.append(
                     np.exp(np.dot(
                    (-im2pi / (n_items * upsample_factor)) *
                    (np.arange(upsampled_region_size[0])[:, None] - ax_offset),
                    (np.fft.ifftshift(np.arange(n_items))[None, :]
                     - n_items // 2))))
            #print('dim_kernels',len(dim_kernels))
            # To compute the upsampled DFT across all spatial dimensions, a tensor product is computed with einsum
            try:
                return np.einsum('ijk, li, mj, nk -> lmn', data, *dim_kernels, optimize=True)
            except TypeError:
           # warnings.warn("Subpixel registration of 3D images will be very slow if your numpy version is earlier than 1.12")
                return np.einsum('ijk, li, mj, nk -> lmn', data, *dim_kernels)

    def _warp_gpu(self, src, shift):

        theta = th.Tensor([[[1, 0, 0, shift[2]*2/self.frame_shape[1]],
                            [0, 1, 0, shift[1]*2/self.frame_shape[2]],
                            [0, 0, 1, shift[0]*2/self.frame_shape[0]]]]).cuda()

        self.affine_grid_size = th.Size([1, 1, self.frame_shape[0], self.frame_shape[2], self.frame_shape[1]])
        grid = self._affine_grid(theta, self.affine_grid_size)
        proj = grid_sample(src.unsqueeze(0).unsqueeze(0), grid)
        return proj.squeeze(0).squeeze(0)

    @staticmethod
    def _affine_grid( theta, size):
        #assert type(size) == torch.Size
        #ctx.size = size
        #ctx.is_cuda = theta.is_cuda
        if len(size) == 5:
            N, C, D, H, W = size
            base_grid = theta.new(N, D, H, W, 4)

            base_grid[:, :, :, :, 0] = (th.linspace(-1, 1, W) if W > 1 else th.Tensor([-1]))
            base_grid[:, :, :, :, 1] = (th.linspace(-1, 1, H) if H > 1 else th.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, :, 2] = (th.linspace(-1, 1, D) if D > 1 else th.Tensor([-1]))\
                .unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 3] = 1

            grid = th.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)
            #ctx.base_grid = base_grid

        return grid

    def _create_queue(self, function, maxsize=0):
        q = Queue(maxsize=maxsize)
        for i in range(self.thread_count):
            t = Thread(target=function, args=(q,))
            t.setDaemon(True)
            t.start()
        return q

pyfish=Pyfish()
displacements, aligned = pyfish.register_whole_stack()
