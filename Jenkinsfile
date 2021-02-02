pipeline {
   options { disableConcurrentBuilds() }
   agent { label 'dss-plugin-tests'}
   environment {
        PLUGIN_INTEGRATION_TEST_INSTANCE="$HOME/instance_config.json"
    }
   stages {
      stage('Run Unit Tests') {
         steps {
            sh 'echo "Running unit tests"'
            catchError(stageResult: 'FAILURE') {
            sh """
               make unit-tests
               """
            }
            sh 'echo "Done with unit tests"'
         }
      }
      stage('Run Integration Tests') {
         steps {
            sh 'echo "Running integration tests"'
            catchError(stageResult: 'FAILURE') {
            sh """
               make integration-tests
               """
            }
            sh 'echo "Done with integration tests"'
         }
      }
   }
   post {
     always {
        script {
           allure([
                    includeProperties: false,
                    jdk: '',
                    properties: [],
                    reportBuildPolicy: 'ALWAYS',
                    results: [[path: 'tests/allure_report']]
            ])

            def status = currentBuild.currentResult
            sh "file_name=\$(echo ${env.JOB_NAME} | tr '/' '-').status; touch \$file_name; echo \"${env.BUILD_URL};${env.CHANGE_TITLE};${env.CHANGE_AUTHOR};${env.CHANGE_URL};${env.BRANCH_NAME};${status};\" >> $HOME/daily-statuses/\$file_name"
        }
     }
   }
}
