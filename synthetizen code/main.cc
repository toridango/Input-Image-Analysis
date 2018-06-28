/****************************************************************************
 **
 ** Copyright (C) 2016 The Qt Company Ltd.
 ** Contact: https://www.qt.io/licensing/
 **
 ** This file is part of the documentation of the Qt Toolkit.
 **
 ** $QT_BEGIN_LICENSE:BSD$
 ** Commercial License Usage
 ** Licensees holding valid commercial Qt licenses may use this file in
 ** accordance with the commercial license agreement provided with the
 ** Software or, alternatively, in accordance with the terms contained in
 ** a written agreement between you and The Qt Company. For licensing terms
 ** and conditions see https://www.qt.io/terms-conditions. For further
 ** information use the contact form at https://www.qt.io/contact-us.
 **
 ** BSD License Usage
 ** Alternatively, you may use this file under the terms of the BSD license
 ** as follows:
 **
 ** "Redistribution and use in source and binary forms, with or without
 ** modification, are permitted provided that the following conditions are
 ** met:
 **   * Redistributions of source code must retain the above copyright
 **     notice, this list of conditions and the following disclaimer.
 **   * Redistributions in binary form must reproduce the above copyright
 **     notice, this list of conditions and the following disclaimer in
 **     the documentation and/or other materials provided with the
 **     distribution.
 **   * Neither the name of The Qt Company Ltd nor the names of its
 **     contributors may be used to endorse or promote products derived
 **     from this software without specific prior written permission.
 **
 **
 ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 ** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 ** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 ** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 ** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 ** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 ** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 **
 ** $QT_END_LICENSE$
 **
 ****************************************************************************/

#include <graphics/openglwindow.h>

#include <QtGui/QGuiApplication>
#include <QtGui/QMatrix4x4>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QScreen>

#include <QtCore/qmath.h>
#include <QtCore/qfile.h>
#include <QtCore/qfileinfo.h>

#include <QOpenGLTexture>
#include <QCommandLineParser>
#include <QFileDialog>

#include <FreeImage.h>

#include <cstdio>
#include <iostream>

#include <sstream>
#include <iomanip>


#define NUM_TEXTURES 4

class ViewerWindow : public OpenGLWindow
{
public:
	ViewerWindow();
	ViewerWindow(const char* backgroundPath, const char* irpvPath, const char* irPath, const char* alphaPath, const char* outPath, bool autoClose, bool isSemantic);

	void initialize() override;
	void render() override;

private:
	void loadTexture(const char *filename, int slot, bool a32 = false);
	void AdjustRatio(int w, int h);

	GLuint m_posAttr;
	GLuint m_colAttr;
	GLuint m_texcAttr;
	GLuint m_matrixUniform;

	const char* m_backgroundPath;
	const char* m_irpvPath;
	const char* m_irPath;
	const char* m_alphaPath;
	const char* m_outPath;

	bool m_saved;
	bool m_autoClose;
	bool m_isSemantic;

	QOpenGLShaderProgram *m_program;	
	QOpenGLTexture		 *m_texture[NUM_TEXTURES];
	int m_frame;
	float m_epsilon; // decimal tolerance for ratio semblance

	unsigned int m_winWidth;
	unsigned int m_winHeight;

	GLuint m_defaultFBO;
	std::vector<float> m_pixelBuffer;
};

ViewerWindow::ViewerWindow()
	: m_program(0)
	, m_frame(0)
{
	m_saved = false;
	m_epsilon = 0.001;

	// PLEASE DO NOT EVER CHANGE THIS TO AN ABSOLUTE PATH !!!
	m_backgroundPath = "../../../resources/images/0000_ini.png";
	m_irpvPath = "../../../resources/images/0000_irpv.png";
	m_irPath = "../../../resources/images/0000_ir.png";
	m_alphaPath = "../../../resources/images/0000_alpha.png";
	m_outPath = "../../../resources/output/0000_output.exr";
	// PLEASE DO NOT EVER CHANGE THIS TO AN ABSOLUTE PATH !!!
}


ViewerWindow::ViewerWindow
	(const char* backgroundPath,
	const char* irpvPath,
	const char* irPath,
	const char* alphaPath,
	const char* outPath,
	bool autoClose,
	bool isSemantic)
	: m_program(0)
	, m_frame(0)
{
	m_saved = false;
	m_epsilon = 0.001;

	m_backgroundPath = backgroundPath;
	m_irpvPath = irpvPath;
	m_irPath = irPath;
	m_alphaPath = alphaPath;
	m_outPath = outPath;
	m_autoClose = autoClose;
	m_isSemantic = isSemantic;
}

void ViewerWindow::AdjustRatio(int w, int h)
{
	float targetRatio = float(w) / float(h);
	resize(w, h);
	bool correct;
	correct = fabs(targetRatio - ( float(width()) / float(height()) )) < m_epsilon;
	while (!correct)
	{
		w = (9 * w) / 10;
		h = (9 * h) / 10;
		resize(w, h);
		correct = fabs(targetRatio - (float(width())/float(height()))) < m_epsilon;
	}


	m_winWidth = width();
	m_winHeight = height();

}


void ViewerWindow::loadTexture(const char *filename, int slot, bool alpha32)
{
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename, 0); //Automatocally detects the format(from over 20 formats!)
	bool isEXR = ((format == FIF_SGI/*Octane EXR*/) || (format == FIF_EXR));
	if (alpha32) std::cout << "The image format is: " << format << std::endl;

	if (!isEXR)
	{
		// simple case with png texture supported by QImage
		m_texture[slot] = new QOpenGLTexture(QImage(filename).mirrored());

		if (slot == 0)
		{
			m_winWidth = m_texture[slot]->width();
			m_winHeight = m_texture[slot]->height();
			std::cout << "Size at load " << m_winWidth << " " << m_winHeight << std::endl;

			AdjustRatio(m_winWidth, m_winHeight);
		}
	}
	else {
		// find the buffer format
		FIBITMAP* image = FreeImage_Load(format, filename);

		// translate from FreeImage to QImage:
		// https://github.com/zoon/FreeImage-Plugin-for-QT4/blob/master/FreeImageHandler.cpp
		int w = FreeImage_GetWidth(image);
		int h = FreeImage_GetHeight(image);
		std::cout << "The size of the image is: " << "(" << format << ")" << w << " * " << h << std::endl; //Some debugging code

		if (slot == 0)
		{
			m_winWidth = w;
			m_winHeight = h;
			std::cout << "Size at load " << m_winWidth << " " << m_winHeight << std::endl;

			AdjustRatio(w, h);

		}


		// standard bitmap type																												
		WORD bpp = FreeImage_GetBPP(image);
		std::cout << "The image is BPP: " << bpp << " alpha32? " << alpha32 << std::endl;

		m_texture[slot] = new QOpenGLTexture(QOpenGLTexture::Target2D);
		m_texture[slot]->setSize(w, h);
		
		QOpenGLTexture::TextureFormat pxintformat;
		QOpenGLTexture::PixelFormat pxformat;
		QOpenGLTexture::PixelType pxtype; 		
		switch (bpp)
		{
			case(128):
			{			// Float - 128 bpp // bitmap type 28
				pxintformat = QOpenGLTexture::TextureFormat::RGBA16_UNorm;
				pxformat = QOpenGLTexture::PixelFormat::RGBA;
				pxtype = QOpenGLTexture::PixelType::Float32;
			} break;
			case(96):
			{			// RGB 
				pxintformat = QOpenGLTexture::TextureFormat::RGBA16_UNorm; /*RGBA8_UNorm*/
				pxformat = QOpenGLTexture::PixelFormat::RGB;
				pxtype = QOpenGLTexture::PixelType::Float32;
			} break;
			case(32):
			{ 
				// Depth
				pxintformat = QOpenGLTexture::TextureFormat::RGBA16_UNorm;
				pxformat = QOpenGLTexture::PixelFormat::Depth;
				pxtype = QOpenGLTexture::PixelType::Float32;
			} break;
			case(24):
			{ 		// RGB 24bit (8bit/channel)
				pxintformat = QOpenGLTexture::TextureFormat::RGBA16_UNorm;
				pxformat = QOpenGLTexture::PixelFormat::RGB;
				pxtype = QOpenGLTexture::PixelType::UInt8;
			} break;
			default: // didn't solve the stuck 4th image problem // left it as failproof
			{
				pxintformat = QOpenGLTexture::TextureFormat::RGBA16_UNorm;
				pxformat = QOpenGLTexture::PixelFormat::RGB;
				pxtype = QOpenGLTexture::PixelType::Float32;
			} break;
		}
		m_texture[slot]->setFormat(pxintformat); // internal format
		m_texture[slot]->allocateStorage(pxformat, pxtype); // format & type
		// Avoid texture blurring ...
		m_texture[slot]->setMinificationFilter(QOpenGLTexture::Filter::Nearest);
		m_texture[slot]->setMagnificationFilter(QOpenGLTexture::Filter::Nearest);


		GLfloat* texture = new GLfloat[4 * w*h];
		GLfloat* depth; if (bpp == 32) depth = new GLfloat[w*h];


		GLfloat* pixels = (GLfloat*)FreeImage_GetBits(image);

		//FreeImage loads in BGR format, so you need to swap some bytes(Or use GL_BGR).

		for (size_t j = 0; j < w*h; j++)
		{

			if (bpp != 32) {

				texture[j * 4 + 0] = pixels[j * 4 + 0];
				texture[j * 4 + 1] = pixels[j * 4 + 1];
				texture[j * 4 + 2] = pixels[j * 4 + 2];
				texture[j * 4 + 3] = pixels[j * 4 + 3];
			}
			else {
				depth[j] = pixels[j];
			}
		}

		m_texture[slot]->setData(pxformat, pxtype, (GLvoid*) (bpp != 32) ? texture : depth);		
	}

	m_program->bind();
	m_program->setUniformValue("tex_background", 0);
	m_program->setUniformValue("tex_irpv", 1);
	m_program->setUniformValue("tex_ir", 2);
	m_program->setUniformValue("tex_alpha", 3);
}

void ViewerWindow::initialize()
{
	m_program = new QOpenGLShaderProgram(this);
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/default.vp");
	if(!m_isSemantic)
		m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/differential_rendering.fp");
	else
		m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/semantic_composition.fp");
	m_program->link();
	m_posAttr = m_program->attributeLocation("posAttr");
	m_texcAttr = m_program->attributeLocation("texcAttr");
	m_matrixUniform = m_program->uniformLocation("matrix");

	// These are the window's original sizes
	m_winWidth = width(); //m_texture[0]->width();
	m_winHeight = height(); //m_texture[0]->height();

	std::cout << "Loading Bg" << std::endl;
	loadTexture(m_backgroundPath, 0);
	std::cout << "Loading irpv" << std::endl;
	loadTexture(m_irpvPath, 1);
	std::cout << "Loading ir" << std::endl;
	loadTexture(m_irPath, 2);
	std::cout << "Loading Alpha" << std::endl;
	loadTexture(m_alphaPath, 3, true /*alpha is 32 bpp?*/);

	m_defaultFBO = GLuint(0);
	
	std::cout << "Size at init " << m_winWidth << " " << m_winHeight << std::endl;

	m_pixelBuffer.resize(m_winWidth*m_winHeight*4);
}

void ViewerWindow::render()
{
	const qreal retinaScale = devicePixelRatio();
	glViewport(0, 0, width() * retinaScale, height() * retinaScale);

	glClear(GL_COLOR_BUFFER_BIT);

	m_program->bind();

	QMatrix4x4 matrix;
	matrix.ortho(-1.f, 1.f, -1.f, 1.f, 0.f, 1.f);
	matrix.translate(0, 0, 0);

	m_program->setUniformValue(m_matrixUniform, matrix);

	GLfloat vertices[] = {
		-1.0f, -1.0f,
		-1.0f,  1.0f,
		 1.0f,  1.0f,
		 1.0f, -1.0f
	};

	GLfloat tex_coords[] = {
		0, 0,
		0, 1,
		1, 1,
		1, 0
	};

	glVertexAttribPointer(m_posAttr, 2, GL_FLOAT, GL_FALSE, 0, vertices);
	glVertexAttribPointer(m_texcAttr, 2, GL_FLOAT, GL_FALSE, 0, tex_coords);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	for (uint i=0; i<NUM_TEXTURES; i++)
		m_texture[i]->bind(i, QOpenGLTexture::TextureUnitReset::ResetTextureUnit);

	glDrawArrays(GL_QUADS, 0, 4);

	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);


	m_program->release();

	++m_frame;


	if (!m_saved)
	{
		std::cout << " - Saving - " << std::endl;


		m_defaultFBO = defaultFramebufferObject();
		std::cout << "Main FBO Id: " << m_defaultFBO << std::endl;

		int w = m_winWidth;
		int h = m_winHeight;

		glBindFramebuffer(GL_READ_FRAMEBUFFER, m_defaultFBO);
		glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT,(GLvoid *) &(m_pixelBuffer[0]));

		FIBITMAP* pBitmap;
		pBitmap = FreeImage_AllocateT(FIT_RGBAF, w, h, /*96*/128);

		unsigned int nPitch = FreeImage_GetPitch(pBitmap);
		std::cout << "Pitch: " << nPitch << std::endl;
		BYTE *pBits = reinterpret_cast<BYTE*>(FreeImage_GetBits(pBitmap));

		for (int y = 0; y < h; ++y)
		{
			float *pPixel = reinterpret_cast<float*>(pBits);
			for (int x = 0; x < w; ++x)
			{
				pPixel[0] = m_pixelBuffer[(y*w + x) * 4 + 0];
				pPixel[1] = m_pixelBuffer[(y*w + x) * 4 + 1];
				pPixel[2] = m_pixelBuffer[(y*w + x) * 4 + 2];
				pPixel[3] = m_pixelBuffer[(y*w + x) * 4 + 3];
				pPixel += /*3*/4;
			}
			pBits += nPitch;
		}


		FreeImage_Save(FIF_SGI/*SGI*//*should be an EXR*/, pBitmap, m_outPath, 0);
		std::cout << m_outPath << std::endl;
		std::cout << " - Saved - " << std::endl;

		// Free resources
		FreeImage_Unload(pBitmap);

		m_saved = true;
		
		if(m_autoClose) close();
	}
}



int main(int argc, char **argv)
{
	QGuiApplication app(argc, argv);

	QCommandLineParser parser;

	QCommandLineOption autoCloseOption("ac", QCoreApplication::translate("main", "Auto-close after render and save"));
	parser.addOption(autoCloseOption);

	QCommandLineOption semanticOption("s", QCoreApplication::translate("main", "Semantic ground truth of virtual object"));
	parser.addOption(semanticOption);

	QCommandLineOption backgroundOpt("bg",
		QCoreApplication::translate("main", "Background of the composition"),
		QCoreApplication::translate("main", "directory"));
	parser.addOption(backgroundOpt);
	//std::cout << "BG Read" << std::endl;

	QCommandLineOption irpvOpt("irpv",
		QCoreApplication::translate("main", "Full render of the virtual part of the composition"),
		QCoreApplication::translate("main", "directory"));
	parser.addOption(irpvOpt);
	//std::cout << "Irpv Read" << std::endl;

	QCommandLineOption irOpt("ir",
		QCoreApplication::translate("main", "Render of the floor of the composition"),
		QCoreApplication::translate("main", "directory"));
	parser.addOption(irOpt);
	//std::cout << "Ir Read" << std::endl;

	QCommandLineOption alphaOpt("a",
		QCoreApplication::translate("main", "Alpha map of the composition"),
		QCoreApplication::translate("main", "directory"));
	parser.addOption(alphaOpt);
	//std::cout << "Alpha Read" << std::endl;

	// Should be always an EXR file right now !!!
	QCommandLineOption outOpt("o",
		QCoreApplication::translate("main", "Output file path"),
		QCoreApplication::translate("main", "directory"));
	parser.addOption(outOpt);
	std::cout << "Output Read" << std::endl;

	parser.process(app);

	bool autoClose = parser.isSet(autoCloseOption);
	bool isSemantic = parser.isSet(semanticOption);

	QString qBackgroundPath = parser.value(backgroundOpt);
	std::string strBackgroundPath = qBackgroundPath.toUtf8().constData();
	const char* backgroundPath = strBackgroundPath.c_str();
	std::cout << "BG Converted: " << backgroundPath << std::endl;

	QString qIrpvPath = parser.value(irpvOpt);
	std::string strIrpvPath = qIrpvPath.toUtf8().constData();
	const char* irpvPath = strIrpvPath.c_str();
	std::cout << "Irpv Converted: " << irpvPath << " opt = " << qIrpvPath.toStdString().c_str() << std::endl;
	
	QString qIrPath = parser.value(irOpt);
	std::string strIrPath = qIrPath.toUtf8().constData();
	const char* irPath = strIrPath.c_str();
	std::cout << "Ir Converted: " << irPath << std::endl;

	QString qAlphaPath = parser.value(alphaOpt);
	std::string strAlphaPath = qAlphaPath.toUtf8().constData();
	const char* alphaPath = strAlphaPath.c_str();
	std::cout << "Alpha Converted: " << alphaPath << std::endl;

	QString qOutPath = parser.value(outOpt);
	std::string strOutPath = qOutPath.toUtf8().constData();
	const char* outPath = strOutPath.c_str();
	std::cout << "Output Converted: " << outPath << std::endl;


	QSurfaceFormat format;
	format.setSamples(16);

	if (!isSemantic)
		std::cout << "Creating composition with paths: " << std::endl
			<< backgroundPath << std::endl
			<< irpvPath << std::endl
			<< irPath << std::endl
			<< alphaPath << std::endl
			<< outPath << std::endl;
	else
		std::cout << "Creating semantic composition with paths: " << std::endl
		<< backgroundPath << std::endl
		<< irpvPath << std::endl
		<< irPath << std::endl
		<< alphaPath << std::endl
		<< outPath << std::endl;

	if (strBackgroundPath.empty() || strIrpvPath.empty() || strIrPath.empty() || strAlphaPath.empty())
	{
		std::cout << "One or more image paths are missing." << std::endl << "Exiting application." << std::endl;
	} else { 
		

		ViewerWindow window(backgroundPath, irpvPath, irPath, alphaPath, outPath, autoClose, isSemantic);


		window.setFormat(format);
		window.resize(640, 480);
		window.show();
	
		window.setAnimating(true);
	
		// Loading and composition starts now
		return app.exec();

		// This is after main loop
	}

}