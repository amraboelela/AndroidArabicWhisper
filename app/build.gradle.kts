plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.compose)
  id("com.chaquo.python")
}

android {
  namespace = "org.amr.arabicwhisper"
  compileSdk = 36

  defaultConfig {
    applicationId = "org.amr.arabicwhisper"
    minSdk = 24
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

    ndk {
      abiFilters += listOf("arm64-v8a")
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }

  kotlinOptions {
    jvmTarget = "11"
  }

  buildFeatures {
    compose = true
    viewBinding = true
  }

  // Remove C++ build - using Python instead
  // externalNativeBuild {
  //   cmake {
  //     path = file("src/main/cpp/CMakeLists.txt")
  //   }
  // }
}

// Chaquopy configuration
chaquopy {
  defaultConfig {
    pip {
      // Install Python dependencies for faster-whisper
      // ctranslate2 is loaded from native .so library
      install("numpy")
      install("tokenizers")
      install("huggingface-hub")
      install("tqdm")
      // Note: av (PyAV) and onnxruntime might not work on Android
      // We'll handle audio with our own code if needed
    }
  }
}

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material3)
  implementation("androidx.appcompat:appcompat:1.6.1")
  // Remove ONNX Runtime - not needed with Python approach
  // implementation("com.microsoft.onnxruntime:onnxruntime-android:1.15.1")
  implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.ui.test.junit4)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
}