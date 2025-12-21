plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
    id("com.google.devtools.ksp")

}

android {
    namespace = "com.fpf.smartscansdk.core"
    compileSdk = 36

    defaultConfig {
        minSdk = 30
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    packaging {
        resources.excludes.addAll(
            listOf(
                "META-INF/LICENSE.md",
                "META-INF/LICENSE-notice.md",
            )
        )
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    lint {
        targetSdk = 34
    }

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
            all {
                it.useJUnitPlatform()
            }
        }
    }

}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
        vendor = JvmVendorSpec.ADOPTIUM
    }
}

dependencies {
    // Expose core-ktx to consumers of core or extensions
    api(libs.androidx.core.ktx)

    // JVM unit tests
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.junit.jupiter.api)
    testRuntimeOnly(libs.junit.jupiter.engine)
    testImplementation(libs.mockk)
    testImplementation(kotlin("test"))

    // Android instrumented tests
    androidTestImplementation(libs.androidx.core)
    androidTestImplementation(libs.androidx.junit.ktx)
    androidTestImplementation(libs.androidx.runner)
    androidTestImplementation(libs.mockk.android)
    androidTestImplementation(libs.kotlinx.coroutines.test)

    androidTestImplementation(libs.androidx.room.runtime)
    androidTestImplementation(libs.androidx.room.ktx)
    androidTestImplementation(libs.androidx.room.testing)
    ksp(libs.androidx.room.compiler)

}

val gitVersion: String by lazy {
    runCatching {
        val proc = ProcessBuilder("git", "describe", "--tags", "--abbrev=0")
            .redirectErrorStream(true)
            .start()
        val out = proc.inputStream.bufferedReader().use { it.readText() }.trim()
        if (proc.waitFor() == 0) out.removePrefix("v") else throw RuntimeException("git failed")
    }.getOrDefault("1.0.0")
}


publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.github.smartscanapp.smartscan-sdk"
            artifactId = "smartscan-${project.name}"
            version = gitVersion

            afterEvaluate {
                from(components["release"])
            }
        }
    }
}
