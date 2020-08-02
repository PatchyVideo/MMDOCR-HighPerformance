/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NV_UFF_PARSER_H
#define NV_UFF_PARSER_H

#include "NvInfer.h"

//!
//! \file NvUffParser.h
//!
//! This is the API for the UFF Parser
//!

// Current supported Universal Framework Format (UFF) version for the parser.
#define UFF_REQUIRED_VERSION_MAJOR 0
#define UFF_REQUIRED_VERSION_MINOR 6
#define UFF_REQUIRED_VERSION_PATCH 9

//!
//! \namespace nvuffparser
//!
//! \brief The TensorRT UFF parser API namespace.
//!
namespace nvuffparser
{

//!
//! \enum UffInputOrder
//! \brief The different possible supported input order.
//!
enum class UffInputOrder : int
{
    kNCHW = 0,  //!< NCHW order.
    kNHWC = 1,  //!< NHWC order.
    kNC = 2     //!< NC order.
};

//!
//! \enum FieldType
//! \brief The possible field types for custom layer.
//!

enum class FieldType : int
{
    kFLOAT = 0,     //!< FP32 field type.
    kINT32 = 1,     //!< INT32 field type.
    kCHAR = 2,      //!< char field type. String for length>1.
    kDIMS = 4,      //!< nvinfer1::Dims field type.
    kDATATYPE = 5,  //!< nvinfer1::DataType field type.
    kUNKNOWN = 6
};

//!
//! \class FieldMap
//!
//! \brief An array of field params used as a layer parameter for plugin layers.
//!
//! The node fields are passed by the parser to the API through the plugin
//! constructor. The implementation of the plugin should parse the contents of
//! the fieldMap as part of the plugin constructor
//!
class TENSORRTAPI FieldMap
{
public:
    const char* name;
    const void* data;
    FieldType type = FieldType::kUNKNOWN;
    int length = 1;

    FieldMap(const char* name, const void* data, const FieldType type, int length = 1) TRTNOEXCEPT;
};

struct FieldCollection
{
    int nbFields;
    const FieldMap* fields;
};

//!
//! \class IPluginFactory
//!
//! \brief Plugin factory used to configure plugins.
//!
class IPluginFactory
{
public:
    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPlugin.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPlugin(const char* layerName) TRTNOEXCEPT = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //! \param fc A collection of FieldMaps used as layer parameters for different plugin layers.
    //!
    //! \see FieldCollection
    //!
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights,
                                            const FieldCollection fc) TRTNOEXCEPT = 0;

    virtual ~IPluginFactory() {}
};

//!
//! \class IPluginFactoryExt
//!
//! \brief Plugin factory used to configure plugins with added support for TRT versioning.
//!
class IPluginFactoryExt : public IPluginFactory
{
public:
    virtual int getVersion() const TRTNOEXCEPT
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPluginExt.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPluginExt(const char* layerName) TRTNOEXCEPT = 0;
};

//!
//! \class IUffParser
//!
//! \brief Class used for parsing models described using the UFF format.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IUffParser
{
public:
    //!
    //! \brief Register an input name of a UFF network with the associated Dimensions.
    //!
    //! \param inputName Input name.
    //! \param inputDims Input dimensions.
    //! \param inputOrder Input order on which the framework input was originally.
    //!
    virtual bool registerInput(const char* inputName, nvinfer1::Dims inputDims, UffInputOrder inputOrder) TRTNOEXCEPT = 0;

    //!
    //! \brief Register an output name of a UFF network.
    //!
    //! \param outputName Output name.
    //!
    virtual bool registerOutput(const char* outputName) TRTNOEXCEPT = 0;

    //!
    //! \brief Parse a UFF file.
    //!
    //! \param file File name of the UFF file.
    //! \param network Network in which the UFFParser will fill the layers.
    //! \param weightsType The type on which the weights will transformed in.
    //!
    virtual bool parse(const char* file,
                       nvinfer1::INetworkDefinition& network,
                       nvinfer1::DataType weightsType=nvinfer1::DataType::kFLOAT) TRTNOEXCEPT = 0;

    //!
    //! \brief Parse a UFF buffer, useful if the file already live in memory.
    //!
    //! \param buffer Buffer of the UFF file.
    //! \param size Size of buffer of the UFF file.
    //! \param network Network in which the UFFParser will fill the layers.
    //! \param weightsType The type on which the weights will transformed in.
    //!
    virtual bool parseBuffer(const char* buffer, std::size_t size,
                             nvinfer1::INetworkDefinition& network,
                             nvinfer1::DataType weightsType=nvinfer1::DataType::kFLOAT) TRTNOEXCEPT = 0;

    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Return Version Major of the UFF.
    //!
    virtual int getUffRequiredVersionMajor() TRTNOEXCEPT = 0;

    //!
    //! \brief Return Version Minor of the UFF.
    //!
    virtual int getUffRequiredVersionMinor() TRTNOEXCEPT = 0;

    //!
    //! \brief Return Patch Version of the UFF.
    //!
    virtual int getUffRequiredVersionPatch() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactory used to create the user defined plugins.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactory.
    //!
    virtual void setPluginFactory(IPluginFactory* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactoryExt used to create the user defined pluginExts.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactoryExt.
    //!
    virtual void setPluginFactoryExt(IPluginFactoryExt* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the namespace used to lookup and create plugins in the network.
    //!
    virtual void setPluginNamespace(const char* libNamespace) TRTNOEXCEPT = 0;

protected:
    virtual ~IUffParser() {}

public:
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) TRTNOEXCEPT = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual nvinfer1::IErrorRecorder* getErrorRecorder() const TRTNOEXCEPT = 0;
};

//!
//! \brief Creates a IUffParser object.
//!
//! \return A pointer to the IUffParser object is returned.
//!
//! \see nvuffparser::IUffParser
//!
TENSORRTAPI IUffParser* createUffParser() TRTNOEXCEPT;

//!
//! \brief Shuts down protocol buffers library.
//!
//! \note No part of the protocol buffers library can be used after this function is called.
//!
TENSORRTAPI void shutdownProtobufLibrary(void) TRTNOEXCEPT;

} // namespace nvuffparser


//!
//! Internal C entry point for creating IUffParser
//! @private
//!
extern "C" TENSORRTAPI void* createNvUffParser_INTERNAL() TRTNOEXCEPT;

#endif /* !NV_UFF_PARSER_H */
