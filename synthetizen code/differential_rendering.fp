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
	
	if (length(irpv-ir) < 0.3)
	{
		gl_FragColor = alpha.a*irpv/*foreground*/ + (1.0 - alpha.a)*(background); 
			
	}else{
		irpv = irpv / (irpv +1.0);
		ir = ir / (ir +1.0);
		irpv.a = 1.0;
		ir.a = 1.0;
		gl_FragColor = alpha.a*irpv/*foreground*/ + (1.0 - alpha.a)*(background + (irpv - ir)); 
	}
}
