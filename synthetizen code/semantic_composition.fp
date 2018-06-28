
varying mediump vec4 texc;
uniform sampler2D tex_background;  // RGB base image
uniform sampler2D tex_irpv; 
uniform sampler2D tex_ir;
uniform sampler2D tex_alpha;


void main() 
{
	
	vec4 background = texture2D(tex_background, texc.st); 
	vec4 irpv = texture2D(tex_irpv, texc.st);
	vec4 ir = texture2D(tex_ir, texc.st);
	vec4 alpha = texture2D(tex_alpha, texc.st);
	
	if (alpha.a > 0.0)
		gl_FragColor = vec4(irpv.r,irpv.g,irpv.b,1);
	else
		gl_FragColor = background; 
	
}
