import{S as r}from"./index-BYqCYF8m.js";import"./imageProcessingFunctions-C06XhpkA.js";import"./helperFunctions-ZaHdJGQJ.js";import"./index-iG-u5S-q.js";const e="imageProcessingPixelShader",i=`varying vec2 vUV;uniform sampler2D textureSampler;
#include<imageProcessingDeclaration>
#include<helperFunctions>
#include<imageProcessingFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
void main(void)
{vec4 result=texture2D(textureSampler,vUV);result.rgb=max(result.rgb,vec3(0.));
#ifdef IMAGEPROCESSING
#ifndef FROMLINEARSPACE
result.rgb=toLinearSpace(result.rgb);
#endif
result=applyImageProcessing(result);
#else
#ifdef FROMLINEARSPACE
result=applyImageProcessing(result);
#endif
#endif
gl_FragColor=result;}`;r.ShadersStore[e]||(r.ShadersStore[e]=i);const o={name:e,shader:i};export{o as imageProcessingPixelShader};
