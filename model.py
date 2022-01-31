from basic import *
from utils import *

class GuideFormer(nn.Module):
    def __init__(self,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32, 16, 8, 4],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, token_mlp='dwc', 
                 downsample=PatchMerging, upsample=PatchExpand, 
                 use_checkpoint=False, **kwargs):
        super(GuideFormer, self).__init__()

        # GuideFormer parameters
        self.num_enc_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.mlp = token_mlp
        self.win_size = win_size

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[3]
        dec_dpr = enc_dpr[::-1]

        # Color branch
        self.rgb_proj_in = InputProj(in_channels=3, out_channels=embed_dim, kernel_size=3, stride=1,
                                     act_layer=nn.GELU)

        self.rgb_encoder_res1 = BasicBlockGeo(inplanes=embed_dim, planes=embed_dim * 2, stride=2, geoplanes=0)
        self.rgb_encoder_res2 = BasicBlockGeo(inplanes=embed_dim * 2, planes=embed_dim * 4, stride=2, geoplanes=0)

        self.rgb_encoder_layer1 = GuideFormerLayer(dim=embed_dim * 4,
                                                out_dim=embed_dim * 4, depth=depths[0],
                                                num_heads=num_heads[0], win_size=win_size,
                                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer, token_mlp=token_mlp,
                                                use_checkpoint=use_checkpoint)
        self.rgb_downsample1 = downsample(embed_dim * 4)
        self.rgb_encoder_layer2 = GuideFormerLayer(dim=embed_dim * 8,
                                                out_dim=embed_dim * 8, depth=depths[1],
                                                num_heads=num_heads[1], win_size=win_size,
                                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer, token_mlp=token_mlp,
                                                use_checkpoint=use_checkpoint)
        self.rgb_downsample2 = downsample(embed_dim * 8)
        self.rgb_encoder_layer3 = GuideFormerLayer(dim=embed_dim * 16,
                                                out_dim=embed_dim * 16, depth=depths[2],
                                                num_heads=num_heads[2], win_size=win_size,
                                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer, token_mlp=token_mlp,
                                                use_checkpoint=use_checkpoint)
        self.rgb_downsample3 = downsample(embed_dim * 16)

        self.rgb_bottleneck = GuideFormerLayer(dim=embed_dim * 32,
                                                    out_dim=embed_dim * 32, depth=depths[3],
                                                    num_heads=num_heads[3], win_size=11,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=conv_dpr,
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)

        self.rgb_up3 = upsample(embed_dim * 32, embed_dim * 16)
        self.rgb_decoder_layer3 = GuideFormerLayer(dim=embed_dim * 16,
                                                out_dim=embed_dim * 16, depth=depths[-3],
                                                num_heads=num_heads[-3], win_size=win_size,
                                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[-3]],
                                                norm_layer=norm_layer, token_mlp=token_mlp,
                                                use_checkpoint=use_checkpoint)
        self.rgb_up2 = upsample(embed_dim * 16, embed_dim * 8)
        self.rgb_decoder_layer2 = GuideFormerLayer(dim=embed_dim * 8,
                                                    out_dim=embed_dim * 8, depth=depths[-2],
                                                    num_heads=num_heads[-2], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[-3:-2]):sum(depths[-3:-1])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)
        self.rgb_up1 = upsample(embed_dim * 8, embed_dim * 4)
        self.rgb_decoder_layer1 = GuideFormerLayer(dim=embed_dim * 4,
                                                    out_dim=embed_dim * 4, depth=depths[-1],
                                                    num_heads=num_heads[-1], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[-3:-1]):sum(depths[-3:])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)

        self.rgb_decoder_deconv2 = deconvbnrelu(in_channels=embed_dim * 4, out_channels=embed_dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_deconv1 = deconvbnrelu(in_channels=embed_dim * 2, out_channels=embed_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = OutputProj(in_channels=embed_dim * 1, out_channels=2, kernel_size=3, stride=1,
                                             norm_layer=nn.BatchNorm2d, act_layer=nn.GELU)

        # Depth branch
        self.depth_proj_in = InputProj(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1,
                                     act_layer=nn.GELU)

        self.depth_encoder_res1 = BasicBlockGeo(inplanes=embed_dim, planes=embed_dim * 2, stride=2, geoplanes=0)
        self.depth_encoder_res2 = BasicBlockGeo(inplanes=embed_dim * 2, planes=embed_dim * 4, stride=2, geoplanes=0)

        self.depth_encoder_layer1 = GuideFormerLayer(dim=embed_dim * 4,
                                                    out_dim=embed_dim * 4, depth=depths[0],
                                                    num_heads=num_heads[0], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)
        self.depth_downsample1 = downsample(embed_dim * 4)
        self.depth_encoder_layer2 = GuideFormerLayer(dim=embed_dim * 8,
                                                    out_dim=embed_dim * 8, depth=depths[1],
                                                    num_heads=num_heads[1], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)
        self.depth_downsample2 = downsample(embed_dim * 8)
        self.depth_encoder_layer3 = GuideFormerLayer(dim=embed_dim * 16,
                                                    out_dim=embed_dim * 16, depth=depths[2],
                                                    num_heads=num_heads[2], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)
        self.depth_downsample3 = downsample(embed_dim * 16)

        self.depth_bottleneck = GuideFormerLayer(dim=embed_dim * 32,
                                                out_dim=embed_dim * 32, depth=depths[3],
                                                num_heads=num_heads[3], win_size=11,
                                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=conv_dpr,
                                                norm_layer=norm_layer, token_mlp=token_mlp,
                                                use_checkpoint=use_checkpoint)

        self.depth_up3 = upsample(embed_dim * 32, embed_dim * 16)
        self.depth_decoder_layer3 = GuideFormerLayer(dim=embed_dim * 16,
                                                    out_dim=embed_dim * 16, depth=depths[-3],
                                                    num_heads=num_heads[-3], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[:depths[-3]],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)
        self.depth_up2 = upsample(embed_dim * 16, embed_dim * 8)
        self.depth_decoder_layer2 = GuideFormerLayer(dim=embed_dim * 8,
                                                    out_dim=embed_dim * 8, depth=depths[-2],
                                                    num_heads=num_heads[-2], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[-3:-2]):sum(depths[-3:-1])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)
        self.depth_up1 = upsample(embed_dim * 8, embed_dim * 4)
        self.depth_decoder_layer1 = GuideFormerLayer(dim=embed_dim * 4,
                                                    out_dim=embed_dim * 4, depth=depths[-1],
                                                    num_heads=num_heads[-1], win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[-3:-1]):sum(depths[-3:])],
                                                    norm_layer=norm_layer, token_mlp=token_mlp,
                                                    use_checkpoint=use_checkpoint)

        self.depth_decoder_deconv2 = deconvbnrelu(in_channels=embed_dim * 4, out_channels=embed_dim * 2, kernel_size=5,
                                                stride=2, padding=2, output_padding=1)
        self.depth_decoder_deconv1 = deconvbnrelu(in_channels=embed_dim * 2, out_channels=embed_dim, kernel_size=5,
                                                stride=2, padding=2, output_padding=1)
        self.depth_decoder_output = OutputProj(in_channels=embed_dim, out_channels=2, kernel_size=3, stride=1,
                                             norm_layer=nn.BatchNorm2d, act_layer=nn.GELU)


        self.rgb2d_attn1 = FusionLayer(dim=embed_dim * 4,
                                        out_dim=embed_dim * 4, depth=depths[0],
                                        num_heads=num_heads[0], win_size=win_size,
                                        mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                        norm_layer=norm_layer, token_mlp=token_mlp,
                                        use_checkpoint=use_checkpoint)
        self.rgb2d_attn2 = FusionLayer(dim=embed_dim * 8,
                                        out_dim=embed_dim * 8, depth=depths[1],
                                        num_heads=num_heads[1], win_size=win_size,
                                        mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                        norm_layer=norm_layer, token_mlp=token_mlp,
                                        use_checkpoint=use_checkpoint)
        self.rgb2d_attn3 = FusionLayer(dim=embed_dim * 16,
                                        out_dim=embed_dim * 16, depth=depths[2],
                                        num_heads=num_heads[2], win_size=win_size,
                                        mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                        norm_layer=norm_layer, token_mlp=token_mlp,
                                        use_checkpoint=use_checkpoint)
        self.rgb2d_attn_bottleneck = FusionLayer(dim=embed_dim * 32,
                                                out_dim=embed_dim * 32, depth=depths[3],
                                                num_heads=num_heads[3], win_size=10,
                                                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=conv_dpr,
                                                norm_layer=norm_layer, token_mlp=token_mlp,
                                                use_checkpoint=use_checkpoint)

        self.softmax = nn.Softmax(dim=1)

        weights_init(self)

    def forward(self, input):
        rgb = input['rgb']
        d = input['d']

        B, C, H, W = d.shape
        H1, W1 = H, W   # 352(320) 1216
        H2, W2 = (H1 + 1) // 2, (W1 + 1) // 2   # 176(160) 608
        H3, W3 = (H2 + 1) // 2, (W2 + 1) // 2   # 88(80) 304
        H4, W4 = (H3 + 1) // 2, (W3 + 1) // 2   # 44(40) 152
        H5, W5 = (H4 + 1) // 2, (W4 + 1) // 2   # 22(20) 76
        H6, W6 = (H5 + 1) // 2, (W5 + 1) // 2  # 11(10) 38

        # Color branch
        rgb_feature = self.rgb_proj_in(rgb)
        rgb_res1 = self.rgb_encoder_res1(rgb_feature)
        rgb_res2 = self.rgb_encoder_res2(rgb_res1)
        rgb_token0 = rgb_res2.flatten(2).transpose(1, 2).contiguous()

        rgb_token1 = self.rgb_encoder_layer1(rgb_token0, (H3, W3))
        rgb_pool1 = self.rgb_downsample1(rgb_token1, (H3, W3))

        rgb_token2 = self.rgb_encoder_layer2(rgb_pool1, (H4, W4))
        rgb_pool2 = self.rgb_downsample2(rgb_token2, (H4, W4))

        rgb_token3 = self.rgb_encoder_layer3(rgb_pool2, (H5, W5))
        rgb_pool3 = self.rgb_downsample3(rgb_token3, (H5, W5))

        rgb_token_bottle = self.rgb_bottleneck(rgb_pool3, (H6, W6))

        rgb_up3 = self.rgb_up3(rgb_token_bottle, (H6, W6), (H5, W5)) + rgb_token3
        rgb_feature_decoder3 = self.rgb_decoder_layer3(rgb_up3, (H5, W5))

        rgb_up2 = self.rgb_up2(rgb_feature_decoder3, (H5, W5), (H4, W4)) + rgb_token2
        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_up2, (H4, W4))

        rgb_up1 = self.rgb_up1(rgb_feature_decoder2, (H4, W4), (H3, W3)) + rgb_token1
        rgb_feature_decoder1 = self.rgb_decoder_layer1(rgb_up1, (H3, W3))

        B, _, C = rgb_feature_decoder1.shape
        rgb_feature_decoder02 = rgb_feature_decoder1.transpose(1, 2).contiguous().view(B, C, H3, W3).contiguous() + rgb_res2
        rgb_feature_decoder02 = self.rgb_decoder_deconv2(rgb_feature_decoder02) + rgb_res1
        rgb_feature_decoder01 = self.rgb_decoder_deconv1(rgb_feature_decoder02)

        rgb_output = self.rgb_decoder_output(rgb_feature_decoder01)
        rgb_depth, rgb_conf = torch.chunk(rgb_output, 2, dim=1)


        ### Depth branch ###
        depth_feature = self.depth_proj_in(d)
        depth_res1 = self.depth_encoder_res1(depth_feature)
        depth_res2 = self.depth_encoder_res2(depth_res1)
        depth_token0 = depth_res2.flatten(2).transpose(1, 2).contiguous()

        depth_token1_cross = self.rgb2d_attn1(depth_token0, rgb_feature_decoder1, (H3, W3))
        depth_token1 = self.depth_encoder_layer1(depth_token1_cross, (H3, W3))
        depth_pool1 = self.depth_downsample1(depth_token1, (H3, W3))

        depth_token2_cross = self.rgb2d_attn2(depth_pool1, rgb_feature_decoder2, (H4, W4))
        depth_token2 = self.depth_encoder_layer2(depth_token2_cross, (H4, W4))
        depth_pool2 = self.depth_downsample2(depth_token2, (H4, W4))

        depth_token3_cross = self.rgb2d_attn3(depth_pool2, rgb_feature_decoder3, (H5, W5))
        depth_token3 = self.depth_encoder_layer3(depth_token3_cross, (H5, W5))
        depth_pool3 = self.depth_downsample3(depth_token3, (H5, W5))

        depth_token_bottle_cross = self.rgb2d_attn_bottleneck(depth_pool3, rgb_token_bottle, (H6, W6))
        depth_token_bottle = self.depth_bottleneck(depth_token_bottle_cross, (H6, W6))

        depth_up3 = self.depth_up3(depth_token_bottle, (H6, W6), (H5, W5)) + depth_token3
        depth_feature_decoder3 = self.depth_decoder_layer3(depth_up3, (H5, W5))

        depth_up2 = self.depth_up2(depth_feature_decoder3, (H5, W5), (H4, W4)) + depth_token2
        depth_feature_decoder2 = self.depth_decoder_layer2(depth_up2, (H4, W4))

        depth_up1 = self.depth_up1(depth_feature_decoder2, (H4, W4), (H3, W3)) + depth_token1
        depth_feature_decoder1 = self.depth_decoder_layer1(depth_up1, (H3, W3))

        B, _, C = depth_feature_decoder1.shape
        depth_feature_decoder02 = depth_feature_decoder1.transpose(1, 2).contiguous().view(B, C, H3, W3).contiguous() + depth_res2
        depth_feature_decoder02 = self.depth_decoder_deconv2(depth_feature_decoder02) + depth_res1
        depth_feature_decoder01 = self.depth_decoder_deconv1(depth_feature_decoder02)

        depth_output = self.depth_decoder_output(depth_feature_decoder01)
        d_depth, d_conf = torch.chunk(depth_output, 2, dim=1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf, d_conf), dim=1)), 2, dim=1)
        output = rgb_conf * rgb_depth + d_conf * d_depth

        return rgb_depth, d_depth, output
