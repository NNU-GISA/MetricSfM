//imagebase.h
/********************************************************************
	imagebase
	created:	2014/10/24
	author:		LX_whu 
	purpose:	This file is for imagebase function
*********************************************************************/
#if !defined imagebase_h__LX_whu_2014_10_24
#define imagebase_h__LX_whu_2014_10_24

#ifdef PACKAGE_GDAL_EXPORTS
#define IMAGEBASE_API __declspec(dllexport)
#else
#define IMAGEBASE_API __declspec(dllimport)

#if defined(_WIN64)  || defined(_X64)

#pragma comment(lib,"package_gdal_x64.lib") 
#pragma message("Automatically linking with package_gdal_x64.lib") 

#else
#ifdef _DEBUG_IMAGEBASE_API
#pragma comment(lib,"package_gdalD.lib") 
#pragma message("Automatically linking with package_gdalD.lib") 
#else
#pragma comment(lib,"package_gdal.lib") 
#pragma message("Automatically linking with package_gdal.lib") 
#endif
#endif

#endif

#ifndef WIN32
typedef 	unsigned int	HWND;
typedef 	unsigned int	UINT;
inline static bool IsWindow(HWND hWnd) { return false; };
inline static void SendMessage(HWND hWnd, UINT Msg, UINT wParam, UINT lParam)
{
	return;
}
#else 
#include <windows.h>
#endif

#define		NOVALUE				-99999

//band col row all start from 0
class IMAGEBASE_API CImageBase {
public:
	CImageBase(void);
	virtual ~CImageBase();

	/************************************************************************/
	/*SS_BIP: BGRBGRBGR......	SS_BSQ:BBBBB......GGGGG......RRRR......     */
	/************************************************************************/
	enum SAMP_STOR{ SS_NONE = 0, SS_BIP = 1, SS_BSQ = 2, };

	enum OPENFLAGS { modeRead = 0x0000,modeReadWrite = 0x0001 };

	bool IsLoaded();
	virtual bool Open(const char* lpstrPath, int mode = modeRead );
	virtual bool Create(const char* lpstrPath, int nCols, int nRows, int nBands, int nBits = 8, SAMP_STOR samp_store = SS_BIP );
	void Close();
	/************************************************************************/
	/* Read/Write: all data is store as BSQ(BBBB...GGGG...RRRR)             */
	/************************************************************************/
	bool Read(void* data, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL);
	bool Write(void* data, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL);
	bool ReadBand(void* data, int band, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL);
	bool WriteBand(void* data, int band, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL);
	bool Read8(BYTE* data, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL);
	bool ReadBand8(BYTE* data, int band, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL);
	
	/*if band < 3 ReadGray8 is same as Read8
	***if band > 3 ReadGray8 use Gray = R*0.299 + G*0.587 + B*0.114 to change RGB to Gray
	*/
	bool ReadGray8(BYTE* data, int stCol, int stRow, int nCols, int nRows, float zoom = 1.0f, int* zoomCols = NULL, int* zoomRows = NULL, bool bBGR = true);

	bool Read(void* data, int stCol, int stRow, int nCols, int nRows, SAMP_STOR samp_store, bool bInvert3Bands = false);
	bool Write(void* data, int stCol, int stRow, int nCols, int nRows, SAMP_STOR samp_store, bool bInvert3Bands = false);

	double	GetBandVal(double col, double row, int band);
	double  GetBandVal4Geo(double x, double y, int band){
		GetImgXY(&x, &y);
		return GetBandVal(x, y, band);
	}

	BYTE	GetBandVal8(int col, int row, int band);

	void	GetGeoXY(double* x,double* y);
	void	GetImgXY(double* x, double* y);

	int	GetRows();
	int GetCols();
	int GetBands();
	int GetBits();
	int GetByte();

	bool	GetNoDataValue(int nBandIdx, double* val);
	bool	SetNoDataValue(int nBandIdx, double val);
	void	SetNoDataValue(double val){
		for (int i = 0; i < GetBands(); i++){
			SetNoDataValue(i, val);
		}
	}

	bool GetStatistics(int nBandIdx, double *pdfMin, double *pdfMax, double *pdfMean, double *padfStdDev);

	void			GetGeoTransform(double * padfTransform, bool bPixelCenter = true);
	const char*		GetProjectionRef();

	/*set GeoTransform
	***Xgeo = padfTransform[0]+padfTransform[1]*col+padfTransform[2]*row
	***Ygeo = padfTransform[3]+padfTransform[4]*col+padfTransform[5]*row
	***the image coordination origin is left upper center
	*/
	bool			SetGeoTransform(double * padfTransform,bool bPixelCenter = true);
	//set projection reference ,you can use img2.SetProjectionRef(img1.GetProjectionRef()) to set it
	bool			SetProjectionRef(const char* lpstrProjectionRef);
	bool			CopyGeoInformation(CImageBase& img);

	const char*		GetImagePath();
protected:
	virtual void Reset();
	bool	malloc_data_buf(int nCols, int nRows, int nBands, int datasize);
	void	calc_buf_size(int* nCols, int* nRows, int datasize);
	bool	adjust_data_buf(double col, double row);
	double	GetBandBufVal(double col, double row, int band);
	void	SetBandBufVal(int col, int row, int band, double val);
	BYTE*	GetBandBuf(int col, int row, int band);
protected:
	class image*	m_pImgSet;
private:
	BYTE*			m_pImgBuf;
	int				m_stBufCol;
	int				m_stBufRow;
	int				m_nBufCols;
	int				m_nBufRows;
};

inline void ApplyGeoTransform(const double *padfGeoTransform,
	double dfPixel, double dfLine,
	double *pdfGeoX, double *pdfGeoY)
{
	*pdfGeoX = padfGeoTransform[0] + dfPixel * padfGeoTransform[1]
		+ dfLine  * padfGeoTransform[2];
	*pdfGeoY = padfGeoTransform[3] + dfPixel * padfGeoTransform[4]
		+ dfLine  * padfGeoTransform[5];
}
bool IMAGEBASE_API InvGeoTransform(const double *gt_in, double *gt_out);

inline static bool SaveImageFile(const char *strImgName, unsigned char *pImg, int cols, int rows, int bands, int bits = 8, double* padfGeoTransform = NULL, const char* lpstrWkt = NULL)
{
	CImageBase img;	if (!img.Create(strImgName, cols, rows, bands, bits)) return false;
	if( !img.Write(pImg, 0, 0, cols, rows) ) return false;
	if (padfGeoTransform) img.SetGeoTransform(padfGeoTransform);
	if (lpstrWkt) img.SetProjectionRef(lpstrWkt);
	img.Close();
	return true;
}

inline static bool ReadImageFile(const char *strImgName, unsigned char *pImg, int cols, int rows, int bands, int bits = 8, double* padfGeoTransform = NULL, const char* lpstrWkt = NULL)
{
	CImageBase img;	if (!img.Open(strImgName)) return false;
	if (!img.Read(pImg, 0, 0, cols, rows)) return false;
	if (padfGeoTransform) img.SetGeoTransform(padfGeoTransform);
	if (lpstrWkt) img.SetProjectionRef(lpstrWkt);
	img.Close();
	return true;
}

#endif // imagebase_h__LX_whu_2014_10_24

