using Images
using StatsBase: sample, shuffle
using DataAugmentation
using Flux
using TestImages
import Base: length, getindex
using BenchmarkTools

const im_size = (224, 224)
imgs = rand(["chelsea", "coffee"], 1000)

struct ImageContainer{T<:Vector}
    img::T
end

length(data::ImageContainer) = length(data.img)
tfm = DataAugmentation.compose(ScaleKeepAspect(im_size))

function getindex(data::ImageContainer, idx::Int)
    path = data.img[idx]
    _img = testimage(path)
    _img = apply(tfm, Image(_img))
    img = collect(channelview(float32.(itemdata(_img))))
    return img
end

data = ImageContainer(imgs)
deval =
    Flux.DataLoader(data, batchsize = 32, parallel = true, collate = true, partial = true)

function data_loop(data)
    count = 0
    for x in data
        count += last(size(x))
    end
    return nothing
end

for i = 1:20
    @time data_loop(deval)
end
# @btime data_loop($deval)
# @btime data_loop($deval)
# @btime data_loop($deval)
